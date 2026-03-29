#include "wavreader/wavreader.h"
#include "miniaudio.h"
#include "signal.h"
#include "calculator.hpp"
#include "loader.hpp"
#include "saver.hpp"
#include "stream.hpp"
#include "onnxer.hpp"
#include "onnxruntime_cxx_api.h"

#include <iostream>
#include <vector>

#include <chrono>
#include <numeric>
#include <algorithm>

using namespace std;

struct ProgramOptions {
    std::string input_file;
    std::vector<std::string> model_paths; // 使用vector存储可变参数
    bool has_i = false;
    bool has_m = false;
};

void print_usage(const char* prog_name) 
{
    std::cerr << "Usage: " << prog_name 
              << " [-i <filename>] [-m <model1> [model2]]\n";
}

bool parse_args(int argc, char** argv, ProgramOptions& opts) 
{
    for (int i = 1; i < argc; ) 
    {
        std::string arg = argv[i];
        
        if (arg == "-i") 
        {
            if (i + 1 >= argc) 
            {
                std::cerr << "Error: -i requires a filename argument\n";
                return false;
            }
            opts.input_file = argv[i+1];
            opts.has_i = true;
            i += 2;
        } 
        else if (arg == "-m") 
        {
            if (i + 1 >= argc) 
            {
                std::cerr << "Error: -m requires at least 1 argument\n";
                return false;
            }
            
            opts.model_paths.push_back(argv[i+1]); // 第一个参数必选
            i += 2;
            
            // 检查是否有第二个可选参数
            if (i < argc && argv[i][0] != '-') 
            { 
                // 下一个不是选项
                opts.model_paths.push_back(argv[i]);
                i += 1;
            }
            opts.has_m = true;
        }
        else 
        {
            std::cerr << "Error: Unknown option " << arg << "\n";
            return false;
        }
    }
    return true;
}


int main(int argc, char* argv[])
{
    ProgramOptions opts;
    if (!parse_args(argc, argv, opts)) 
    {
        print_usage(argv[0]);
        return 1;
    }

    int nfft = 512, hop_len = nfft/2;
    if(opts.has_m && opts.has_i && opts.model_paths.size()==1)
    {
        SE::Calculator calculator(nfft, hop_len, opts.model_paths[0]);
        AudioStreamer streamer(opts.input_file, nfft, hop_len, 16000, &calculator, true); 
        streamer.start();
        std::cout << "press any key to exit..." << std::endl;
        std::string tmp;
        cin >> tmp;
        streamer.stop();
        return 0;
    }
    else if(opts.has_m && opts.has_i && opts.model_paths.size()==2)
    {
        SE::Calculator calculator(nfft, hop_len, opts.model_paths[0], opts.model_paths[1]);
        AudioStreamer streamer(opts.input_file, nfft, hop_len, 16000, &calculator, true); 
        streamer.start();
        std::cout << "press any key to exit..." << std::endl;
        std::string tmp;
        cin >> tmp;
        streamer.stop();
        return 0;
    }
    else if(opts.has_m && !opts.has_i && opts.model_paths.size()==1)
    {
        SE::Calculator calculator(nfft, hop_len, opts.model_paths[0]);
        AudioStreamer streamer(nfft, hop_len, 16000, &calculator, true); 
        streamer.start();
        std::cout << "press any key to exit..." << std::endl;
        std::string tmp;
        cin >> tmp;
        streamer.stop();
        return 0;
    }
    else if(opts.has_m && !opts.has_i && opts.model_paths.size()==2)
    {
        SE::Calculator calculator(nfft, hop_len, opts.model_paths[0], opts.model_paths[1]);
        AudioStreamer streamer(nfft, hop_len, 16000, &calculator, true); 
        streamer.start();
        std::cout << "press any key to exit..." << std::endl;
        std::string tmp;
        cin >> tmp;
        streamer.stop();
        return 0;
    }
    else if(!opts.has_m && opts.has_i)
    {
        SE::Calculator calculator(nfft, hop_len);
        AudioStreamer streamer(opts.input_file, nfft, hop_len, 16000, &calculator, true); 
        streamer.start();
        std::cout << "press any key to exit..." << std::endl;
        std::string tmp;
        cin >> tmp;
        streamer.stop();
        return 0;
    }
    else if(!opts.has_m && !opts.has_i)
    {
        SE::Calculator calculator(nfft, hop_len);
        AudioStreamer streamer(nfft, hop_len, 16000, &calculator, true); 
        streamer.start();
        std::cout << "press any key to exit..." << std::endl;
        std::string tmp;
        cin >> tmp;
        streamer.stop();
        return 0;
    }
    else
    {
        print_usage(argv[0]);
        return -1;
    }
}

