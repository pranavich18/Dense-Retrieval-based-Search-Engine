#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cctype>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <cwctype>
#include <locale>

using namespace std;

//Tokenize sentences into terms
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

// Write sorted postings to a binary run file
void write_file(const vector<tuple<string, int, int>>& postings, int runNumber) {
    string filename = "partial/run_" + to_string(runNumber) + ".bin";
    ofstream out(filename, ios::binary);
    
    for (const auto& [term, docID, freq] : postings) {
        // Write term length and term
        int termLen = term.size();
        out.write(reinterpret_cast<const char*>(&termLen), sizeof(int));
        out.write(term.data(), termLen);
        
        // Write docID and freq
        out.write(reinterpret_cast<const char*>(&docID), sizeof(int));
        out.write(reinterpret_cast<const char*>(&freq), sizeof(int));
    }
    
    out.close();
    
    cerr << "Saved partial index: run_" << runNumber << ".bin ("<< postings.size() << " postings)\n";
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_file.tsv>\n";
        return 1;
    }


    // Create directories
    system("mkdir -p partial");
    system("mkdir -p index");

    string inputFilename = argv[1];
    ifstream file(inputFilename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file '" << inputFilename << "'\n";
        return 1;
    }
    unordered_set<string> allowedPIDs;

    // read subset of passage ids to filter out which opassages to index
    ifstream subsetFile("msmarco_passages_subset.tsv");
    string id;
    while (getline(subsetFile, id)) {
        id.erase(remove_if(id.begin(), id.end(), ::isspace), id.end());
        if (!id.empty())
            allowedPIDs.insert(id);
    }
    cerr << "Loaded " << allowedPIDs.size() << " allowed passage IDs\n";

    // Buffer for postings
    vector<tuple<string, int, int>> postingBuffer;
    const size_t MAX_BUFFER_SIZE = 10000000;
    
    int docID = 0;
    int runNumber = 0;
    string line;
    
    // Document metadata
    ofstream pageTable("index/page_table.txt");
    ofstream docLengthFile("index/doc_lengths.txt");

    ofstream docStoreFile("index/documents.dat", ios::binary);
    ofstream docStoreIndexFile("index/documents.idx", ios::binary);

    cerr << "Starting indexing...\n";

    while (getline(file, line)) {
        stringstream ss(line);
        string pid_str, passage;
        
        if (!getline(ss, pid_str, '\t')) continue;
        if (!getline(ss, passage)) continue;

        // if the passage id is not part of the subset, we skip it
        if (allowedPIDs.find(pid_str) == allowedPIDs.end()) continue;

        long long offset = docStoreFile.tellp();
        int length = passage.length();
        docStoreFile.write(passage.c_str(), length);
        docStoreIndexFile.write(reinterpret_cast<const char*>(&offset), sizeof(long long));
        docStoreIndexFile.write(reinterpret_cast<const char*>(&length), sizeof(int));

        // Tokenize passage
        vector<string> tokens = tokenize(passage);
        if (tokens.empty()) continue;

        // Write to page table
        pageTable << docID << "\t" << pid_str << "\n";
        
        // Write document length
        docLengthFile << docID << "\t" << tokens.size() << "\n";

        // Count term frequencies
        unordered_map<string, int> termFreq;
        for (const auto& t : tokens) {
            termFreq[t]++;
        }

        // Add postings to buffer
        for (const auto& pair : termFreq) {
            postingBuffer.push_back({pair.first, docID, pair.second});
        }

        docID++;

        // Progress update every 100k documents
        if (docID % 100000 == 0) {
            cerr << "Indexed " << docID << " documents...\n";
        }

        // Flush buffer when full
        if (postingBuffer.size() >= MAX_BUFFER_SIZE) {
            sort(postingBuffer.begin(), postingBuffer.end());
            write_file(postingBuffer, runNumber++);
            postingBuffer.clear();
        }
    }

    file.close();
    pageTable.close();
    docLengthFile.close();
    docStoreFile.close();
    docStoreIndexFile.close();
   
    if (!postingBuffer.empty()) {
        sort(postingBuffer.begin(), postingBuffer.end());
        write_file(postingBuffer, runNumber++);
    }

    // Write metadata
    ofstream metaFile("index/indexer_meta.txt");
    metaFile << "total_documents\t" << docID << "\n";
    metaFile << "total_runs\t" << runNumber << "\n";
    metaFile.close();

    cerr << "\nIndexing complete!\n";
    cerr << "Total documents: " << docID << "\n";
    cerr << "Total runs: " << runNumber << "\n";

    return 0;
}