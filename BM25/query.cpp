#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>

using namespace std;

const double K1 = 1.2;
const double B = 0.75;
const int BLOCK_SIZE = 128;

unordered_map<string, tuple<long long,int,int,int>> lexicon;
vector<int> lastDocIDs, docIDSizes, freqSizes;
unordered_map<int,int> docLengths;
int totalDocuments = 0;
double avgDocLength = 0.0;

mutex mergeMutex;

int varbyte_decode(const unsigned char* data, int& offset) {
    int num = 0, shift = 0;
    unsigned char byte;
    do {
        byte = data[offset++];
        num |= (byte & 0x7F) << shift;
        shift += 7;
    } while (byte & 0x80);
    return num;
}

vector<string> tokenize(const string& s) {
    vector<string> tokens; string tok;
    for(char c : s){
        if(isalnum(c)) tok += tolower(c);
        else if(!tok.empty()) tokens.push_back(tok), tok.clear();
    }
    if(!tok.empty()) tokens.push_back(tok);
    return tokens;
}

class InvertedList {
private:
    ifstream invFile;
    long long startOffset;
    int startBlock, numPostings;
    int currentBlockIdx, positionInBlock;
    vector<int> docIDs, freqs;
    bool finished = false;

    void loadBlock(){
        docIDs.clear(); freqs.clear();

        if(currentBlockIdx >= startBlock + (numPostings + BLOCK_SIZE - 1) / BLOCK_SIZE){
            finished = true; return;
        }

        long long offset = startOffset;
        for(int i = startBlock; i < currentBlockIdx; i++)
            offset += sizeof(int) + docIDSizes[i] + sizeof(int) + freqSizes[i];

        invFile.seekg(offset);

        int docSize; invFile.read((char*)&docSize, sizeof(int));
        vector<unsigned char> d(docSize);
        invFile.read((char*)d.data(), docSize);

        int freqSize; invFile.read((char*)&freqSize, sizeof(int));
        vector<unsigned char> f(freqSize);
        invFile.read((char*)f.data(), freqSize);

        int p = 0; while(p < docSize) docIDs.push_back(varbyte_decode(d.data(), p));
        for(size_t i = 1; i < docIDs.size(); i++) docIDs[i] += docIDs[i-1];

        p = 0; while(p < freqSize) freqs.push_back(varbyte_decode(f.data(), p));

        positionInBlock = 0;
    }

public:
    InvertedList(const string& term){
        auto it = lexicon.find(term);
        if(it == lexicon.end()){ finished = true; return; }

        invFile.open("index/inverted_index.bin", ios::binary);
        startOffset = get<0>(it->second);
        startBlock  = get<1>(it->second);
        numPostings = get<2>(it->second);

        currentBlockIdx = startBlock;
        loadBlock();
    }

    bool nextGEQ(int target){
        if(finished) return false;
        while(currentBlockIdx < startBlock + (numPostings + BLOCK_SIZE - 1) / BLOCK_SIZE) {
            if(lastDocIDs[currentBlockIdx] >= target) {
                if(docIDs.empty() || positionInBlock >= (int)docIDs.size())
                    loadBlock();
                break;
            }
            currentBlockIdx++;
        }
        if(finished) return false;

        while(positionInBlock < (int)docIDs.size()) {
            if(docIDs[positionInBlock] >= target) return true;
            positionInBlock++;
        }
        currentBlockIdx++;
        loadBlock();
        return nextGEQ(target);
    }

    bool hasNext(){ return !finished && positionInBlock < (int)docIDs.size(); }
    int doc(){ return docIDs[positionInBlock]; }
    int freq(){ return freqs[positionInBlock]; }
    void next(){ positionInBlock++; }
};

double BM25(int tf, int dl, int df){
    double idf = log((totalDocuments - df + 0.5) / (df + 0.5));
    return idf * ((tf*(K1+1)) / (tf + K1*(1-B + B*(dl/avgDocLength))));
}

vector<pair<int,double>> processQuery(const vector<string>& terms){
    static thread_local vector<double> score(totalDocuments,0);
    static thread_local vector<int> touched;

    for(const string& term : terms){
        auto it = lexicon.find(term);
        if(it == lexicon.end()) continue;

        int df = get<3>(it->second);
        InvertedList list(term);
        list.nextGEQ(0);

        while(list.hasNext()){
            int d = list.doc(), f = list.freq();
            if(score[d] == 0) touched.push_back(d);
            score[d] += BM25(f, docLengths[d], df);
            list.next();
        }
    }

    vector<pair<int,double>> res;
    for(int d : touched) res.emplace_back(d, score[d]);
    touched.clear();
    for(auto& p : res) score[p.first] = 0;

    sort(res.begin(), res.end(), [](auto&a, auto&b){return a.second>b.second;});
    if(res.size() > 100) res.resize(100);
    return res;
}

bool loadIndex(){
    ifstream f("index/lexicon.txt");
    if(!f) return false;
    string t; long long off; int sb,np,df;
    while(f>>t>>off>>sb>>np>>df) lexicon[t]={off,sb,np,df};
    f.close();

    ifstream m("index/metadata.bin",ios::binary);
    int n; m.read((char*)&n,sizeof(int));
    lastDocIDs.resize(n); docIDSizes.resize(n); freqSizes.resize(n);
    m.read((char*)lastDocIDs.data(),n*4);
    m.read((char*)docIDSizes.data(),n*4);
    m.read((char*)freqSizes.data(),n*4);
    m.close();

    ifstream d("index/doc_lengths.txt");
    int id,l;
    while(d>>id>>l){ docLengths[id]=l; avgDocLength+=l; totalDocuments++; }
    avgDocLength/=totalDocuments;
    return true;
}

int main(){
    if(!loadIndex()){ cerr<<"Index load failed\n"; return 1; }

    ifstream q("queries.dev.tsv");
    vector<string> lines; string line;
    while(getline(q,line)) lines.push_back(line);
    q.close();

    ofstream out("results.txt");

    atomic<size_t> idx(0);
    unsigned T = thread::hardware_concurrency();
    vector<thread> thr;

    auto worker = [&](){
        vector<string> local;
        while(true){
            size_t i = idx++;
            if(i>=lines.size()) break;

            stringstream ss(lines[i]);
            string id,text; getline(ss,id,'\t'); getline(ss,text);
            auto terms = tokenize(text);
            auto res = processQuery(terms);

            int rank=1;
            for(auto& p:res)
                local.push_back(id+" Q0 "+to_string(p.first)+" "+to_string(rank++)+" "+to_string(p.second)+" bm25");
        }

        lock_guard<mutex> lock(mergeMutex);
        for(auto&s:local) out<<s<<"\n";
    };

    for(unsigned i=0;i<T;i++) thr.emplace_back(worker);
    for(auto& t:thr) t.join();

    cout<<"Done.\n";
}
