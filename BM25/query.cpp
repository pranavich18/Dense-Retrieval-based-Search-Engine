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

using namespace std;

const double K1 = 1.2;
const double B = 0.75;
const int BLOCK_SIZE = 128;

unordered_map<string, tuple<long long, int, int, int>> lexicon;
vector<int> lastDocIDs, docIDSizes, freqSizes;
unordered_map<int, int> docLengths;
int totalDocuments = 0;
double avgDocLength = 0.0;

// Thread-safe result writing
mutex resultsMutex;


// Varbyte decoding
int varbyte_decode(const unsigned char* data, int& offset) {
    int num = 0;
    int shift = 0;
    unsigned char byte;
    
    do {
        byte = data[offset++];
        num |= (byte & 0x7F) << shift;
        shift += 7;
    } while (byte & 0x80);
    
    return num;
}

// Tokenize
vector<string> tokenize(const string& s) {
    vector<string> tokens;
    string token;
    for (char c : s) {
        if (isalnum(c)) {
            token += tolower(c);
        } else if (!token.empty()) {
            tokens.push_back(token);
            token.clear();
        }
    }
    if (!token.empty()) {
        tokens.push_back(token);
    }
    return tokens;
}

// Inverted List API
class InvertedList {
private:
    ifstream& invFile;
    string term;
    long long startOffset;
    int startBlock;
    int numPostings;
    
    int currentBlockIdx;
    vector<int> currentDocIDs;
    vector<int> currentFreqs;
    int positionInBlock;
    bool finished;
    
    void decompressBlock(int blockIdx) {
        currentDocIDs.clear();
        currentFreqs.clear();
        
        if (blockIdx >= startBlock + (numPostings + BLOCK_SIZE - 1) / BLOCK_SIZE) {
            finished = true;
            return;
        }
        
        // Calculate file position
        long long offset = startOffset;
        for (int i = startBlock; i < blockIdx; i++) {
            offset += sizeof(int) + docIDSizes[i] + sizeof(int) + freqSizes[i];
        }
        
        invFile.seekg(offset);
        
        // Read docID block
        int docIDSize;
        invFile.read(reinterpret_cast<char*>(&docIDSize), sizeof(int));
        
        vector<unsigned char> docIDBlock(docIDSize);
        invFile.read(reinterpret_cast<char*>(docIDBlock.data()), docIDSize);
        
        // Read freq block
        int freqSize;
        invFile.read(reinterpret_cast<char*>(&freqSize), sizeof(int));
        
        vector<unsigned char> freqBlock(freqSize);
        invFile.read(reinterpret_cast<char*>(freqBlock.data()), freqSize);
        
        // Decode docIDs
        int offset_docID = 0;
        while (offset_docID < docIDSize) {
            currentDocIDs.push_back(varbyte_decode(docIDBlock.data(), offset_docID));
        }
        
        // Convert deltas to absolute
        for (size_t i = 1; i < currentDocIDs.size(); i++) {
            currentDocIDs[i] += currentDocIDs[i-1];
        }
        
        // Decode frequencies
        int offset_freq = 0;
        while (offset_freq < freqSize) {
            currentFreqs.push_back(varbyte_decode(freqBlock.data(), offset_freq));
        }
        
        positionInBlock = 0;
    }
    
public:
    InvertedList(ifstream& file, const string& t) 
        : invFile(file), term(t), finished(false) {
        
        auto it = lexicon.find(term);
        if (it == lexicon.end()) {
            finished = true;
            return;
        }
        
        startOffset = get<0>(it->second);
        startBlock = get<1>(it->second);
        numPostings = get<2>(it->second);
        
        currentBlockIdx = startBlock;
        decompressBlock(currentBlockIdx);
    }
    
    bool nextGEQ(int targetDocID) {
        if (finished) return false;
        
        // Skip blocks using metadata
        while (currentBlockIdx < startBlock + (numPostings + BLOCK_SIZE - 1) / BLOCK_SIZE) {
            if (lastDocIDs[currentBlockIdx] >= targetDocID) {
                if (currentDocIDs.empty() || positionInBlock >= (int)currentDocIDs.size()) {
                    decompressBlock(currentBlockIdx);
                }
                break;
            }
            currentBlockIdx++;
        }
        
        if (finished) return false;
        
        // Find within block
        while (positionInBlock < (int)currentDocIDs.size()) {
            if (currentDocIDs[positionInBlock] >= targetDocID) {
                return true;
            }
            positionInBlock++;
        }
        
        // Move to next block
        currentBlockIdx++;
        if (currentBlockIdx >= startBlock + (numPostings + BLOCK_SIZE - 1) / BLOCK_SIZE) {
            finished = true;
            return false;
        }
        
        decompressBlock(currentBlockIdx);
        return nextGEQ(targetDocID);
    }
    
    bool hasNext() {
        return !finished && positionInBlock < (int)currentDocIDs.size();
    }
    
    int getDocID() {
        if (!hasNext()) return -1;
        return currentDocIDs[positionInBlock];
    }
    
    int getFrequency() {
        if (!hasNext()) return 0;
        return currentFreqs[positionInBlock];
    }
    
    void next() {
        positionInBlock++;
        if (positionInBlock >= (int)currentDocIDs.size()) {
            currentBlockIdx++;
            if (currentBlockIdx < startBlock + (numPostings + BLOCK_SIZE - 1) / BLOCK_SIZE) {
                decompressBlock(currentBlockIdx);
            } else {
                finished = true;
            }
        }
    }
};

// BM25 score
double calculateBM25(int tf, int docLength, int df, int N) {
    double idf = log((N - df + 0.5) / (df + 0.5));
    double tfComponent = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * (docLength / avgDocLength)));
    return idf * tfComponent;
}

vector<pair<int, double>> processDisjunctiveQuery(ifstream& invFile, const vector<string>& queryTerms) {
    const int TOP_K_RESULTS = 100;
    using DocScore = pair<double, int>;
    auto cmp = [](const DocScore &a, const DocScore &b){ return a.first > b.first; };
    priority_queue<DocScore, vector<DocScore>, decltype(cmp)> topHeap(cmp);

    unordered_map<int, double> docScores;

    for (const auto& term : queryTerms) {
        auto it = lexicon.find(term);
        if (it == lexicon.end()) continue;

        int df = get<3>(it->second);
        InvertedList list(invFile, term);
        list.nextGEQ(0);

        while (list.hasNext()) {
            int docID = list.getDocID();
            int freq = list.getFrequency();
            int docLen = docLengths.count(docID) ? docLengths[docID] : (int)avgDocLength;

            double score = calculateBM25(freq, docLen, df, totalDocuments);
            docScores[docID] += score;
            double currentScore = docScores[docID];

            if ((int)topHeap.size() < TOP_K_RESULTS) {
                topHeap.push({currentScore, docID});
            } else if (currentScore > topHeap.top().first) {
                topHeap.pop();
                topHeap.push({currentScore, docID});
            }

            list.next();
        }
    }

    vector<pair<int, double>> results;
    while (!topHeap.empty()) {
        results.push_back({topHeap.top().second, topHeap.top().first});
        topHeap.pop();
    }
    reverse(results.begin(), results.end());
    return results;
}

// ===== Worker for each query
void processQueryLine(const string &line, ofstream &resultsFile) {
    stringstream ss(line);
    string queryIDStr, queryText;
    if (!getline(ss, queryIDStr, '\t')) return;
    if (!getline(ss, queryText)) return;

    int queryID = stoi(queryIDStr);
    vector<string> queryTerms = tokenize(queryText);
    if (queryTerms.empty()) return;

    ifstream invFile("index/inverted_index.bin", ios::binary); // each thread opens its own stream
    if (!invFile.is_open()) return;

    vector<pair<int, double>> results = processDisjunctiveQuery(invFile, queryTerms);
    invFile.close();

    // Write results thread-safely
    lock_guard<mutex> lock(resultsMutex);
    int rank = 1;
    for (auto &p : results) {
        resultsFile << queryID << " Q0 " 
                    << p.first << " " 
                    << rank++ << " "
                    << p.second << " bm25\n";
    }
}

// Conjunctive query
vector<pair<int, double>> processConjunctiveQuery(ifstream& invFile, const vector<string>& queryTerms) {
    if (queryTerms.empty()) return {};
    
    struct TermInfo {
        string term;
        int df;
        InvertedList* list;
    };
    
    vector<TermInfo> termInfos;
    
    // Step 1: Build InvertedLists and store df (document frequency)
    for (const auto& term : queryTerms) {
        auto it = lexicon.find(term);
        if (it == lexicon.end()) {
            for (auto& t : termInfos) delete t.list;
            return {};
        }
        
        int df = get<3>(it->second); // document frequency from lexicon
        termInfos.push_back({term, df, new InvertedList(invFile, term)});
    }
    
    // Step 2: Sort by ascending df (shortest posting list first)
    sort(termInfos.begin(), termInfos.end(),
         [](const TermInfo& a, const TermInfo& b) { return a.df < b.df; });
    
    // Step 3: Extract pointers and dfs in sorted order
    vector<InvertedList*> lists;
    vector<int> dfs;
    for (auto& t : termInfos) {
        lists.push_back(t.list);
        dfs.push_back(t.df);
    }

    unordered_map<int, double> docScores;
    
    // Step 4: Begin traversal from the *shortest* list
    lists[0]->nextGEQ(0);
    while (lists[0]->hasNext()) {
        int docID = lists[0]->getDocID();
        bool inAll = true;
        vector<int> freqs = {lists[0]->getFrequency()};
        
        // Step 5: Intersect with the rest of the lists
        for (size_t i = 1; i < lists.size(); i++) {
            lists[i]->nextGEQ(docID);
            if (!lists[i]->hasNext() || lists[i]->getDocID() != docID) {
                inAll = false;
                break;
            }
            freqs.push_back(lists[i]->getFrequency());
        }
        
        // Step 6: Score documents that appear in all lists
        if (inAll) {
            int docLen = docLengths.count(docID) ? docLengths[docID] : (int)avgDocLength;
            double totalScore = 0.0;
            for (size_t i = 0; i < lists.size(); i++) {
                totalScore += calculateBM25(freqs[i], docLen, dfs[i], totalDocuments);
            }
            docScores[docID] = totalScore;
        }
        
        lists[0]->next(); // move shortest list forward
    }
    
    // Step 7: Cleanup and sort results
    for (auto* list : lists) delete list;
    
    vector<pair<int, double>> results;
    for (const auto& p : docScores) results.emplace_back(p.first, p.second);
    
    sort(results.begin(), results.end(),
         [](const auto& a, const auto& b) { return a.second > b.second; });
    
    return results;
}


// Load index
bool loadIndex() {
    ifstream lexFile("index/lexicon.txt");
    if (!lexFile.is_open()) return false;
    
    string line;
    while (getline(lexFile, line)) {
        stringstream ss(line);
        string term;
        long long offset;
        int startBlock, numPostings, df;
        ss >> term >> offset >> startBlock >> numPostings >> df;
        lexicon[term] = make_tuple(offset, startBlock, numPostings, df);
    }
    lexFile.close();
    
    ifstream metaFile("index/metadata.bin", ios::binary);
    if (!metaFile.is_open()) return false;
    
    int numBlocks;
    metaFile.read(reinterpret_cast<char*>(&numBlocks), sizeof(int));
    
    lastDocIDs.resize(numBlocks);
    docIDSizes.resize(numBlocks);
    freqSizes.resize(numBlocks);
    
    metaFile.read(reinterpret_cast<char*>(lastDocIDs.data()), numBlocks * sizeof(int));
    metaFile.read(reinterpret_cast<char*>(docIDSizes.data()), numBlocks * sizeof(int));
    metaFile.read(reinterpret_cast<char*>(freqSizes.data()), numBlocks * sizeof(int));
    metaFile.close();
    
    ifstream docLenFile("index/doc_lengths.txt");
    if (docLenFile.is_open()) {
        while (getline(docLenFile, line)) {
            stringstream ss(line);
            int docID, length;
            ss >> docID >> length;
            docLengths[docID] = length;
            totalDocuments++;
            avgDocLength += length;
        }
        docLenFile.close();
        avgDocLength /= totalDocuments;
    }
    
    return true;
}

int main() {
    if (!loadIndex()) {
        cerr << "Error loading index\n";
        return 1;
    }

    ifstream queryFile("queries.dev.tsv");
    if (!queryFile.is_open()) {
        cerr << "Error opening queries.dev.tsv\n";
        return 1;
    }

    ofstream resultsFile("results.txt");
    if (!resultsFile.is_open()) {
        cerr << "Error creating results.txt\n";
        return 1;
    }

    vector<string> queryLines;
    string line;
    bool firstLine = true;
    while (getline(queryFile, line)) {
        if (firstLine) { firstLine = false; continue; } // skip header
        queryLines.push_back(line);
    }
    queryFile.close();

    // Multi-threading: one thread per CPU core
    unsigned int nThreads = thread::hardware_concurrency();
    vector<thread> threads;
    atomic<size_t> idx(0);

    auto worker = [&]() {
        while (true) {
            size_t i = idx++;
            if (i >= queryLines.size()) break;
            processQueryLine(queryLines[i], resultsFile);
        }
    };

    for (unsigned int t = 0; t < nThreads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto &th : threads) th.join();

    resultsFile.close();
    cout << "Top 100 BM25 results for all queries written using " << nThreads << " threads.\n";
    return 0;
}