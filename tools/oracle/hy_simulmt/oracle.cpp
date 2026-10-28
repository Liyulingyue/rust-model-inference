// Public APIs only; pinned to lihy715/llama.cpp.Hyv3Speech e41ca421.
#include "llama.h"
#include "mtmd.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>
using json = nlohmann::ordered_json;

struct Trace {
    std::string dir;
    std::map<std::string,int> counts;
    void dump(const std::string & name, const float * data, size_t n, json shape) {
        int occurrence=counts[name]++;
        std::string path=dir+"/"+name+"."+std::to_string(occurrence)+".f32";
        std::ofstream f(path,std::ios::binary); f.write((const char *)data,n*sizeof(float));
        if (!f) throw std::runtime_error("trace write failed");
        std::ofstream(dir+"/trace.jsonl",std::ios::app) << json{{"name",name},{"shape",shape},{"occurrence",occurrence},{"binary_path",path}} << '\n';
    }
};
static bool callback(ggml_tensor * t, bool ask, void * opaque) {
    std::string name=ggml_get_name(t);
    if (name!="enc_inp" && name!="after_transformer" && name!="after_proj2" && name!="projected") return false;
    if (ask) return true;
    if (t->type!=GGML_TYPE_F32 || !ggml_is_contiguous(t)) throw std::runtime_error("non-contiguous checkpoint");
    std::vector<float> data(ggml_nelements(t)); ggml_backend_tensor_get(t,data.data(),0,ggml_nbytes(t));
    json shape=json::array(); for(int i=ggml_n_dims(t)-1;i>=0;i--) shape.push_back(t->ne[i]);
    ((Trace *)opaque)->dump("hy."+name,data.data(),data.size(),shape);
    return true;
}
static std::vector<llama_token> tokenize(const llama_vocab * v,const std::string & s,bool add=false,bool parse=true) {
    std::vector<llama_token> t(s.size()+32);
    int n=llama_tokenize(v,s.data(),s.size(),t.data(),t.size(),add,parse);
    if(n<0) { t.resize(-n); n=llama_tokenize(v,s.data(),s.size(),t.data(),t.size(),add,parse); }
    if(n<0) throw std::runtime_error("tokenize failed"); t.resize(n); return t;
}
static void projections(const char * model,const char * manifest,Trace & trace,int threads) {
    auto * gguf=gguf_init_from_file(model,{true,nullptr});
    if(!gguf) throw std::runtime_error("GGUF metadata load failed");
    std::ifstream weights(model,std::ios::binary), fixture(manifest); json cases; fixture>>cases;
    for(const auto & c:cases) {
        std::string name=c["tensor"]; int expert=c["expert"];
        int64_t id=gguf_find_tensor(gguf,name.c_str()); if(id<0) throw std::runtime_error("missing projection");
        auto * ne=gguf_get_tensor_ne(gguf,id); auto type=gguf_get_tensor_type(gguf,id);
        if(expert<0 || expert>=ne[2] || ne[3]!=1 || (type!=GGML_TYPE_Q4_K && type!=GGML_TYPE_Q6_K && type!=GGML_TYPE_F32))
            throw std::runtime_error("unsupported projection probe");
        auto * ctx=ggml_init({128*1024*1024,nullptr,false}); ggml_cpu_init();
        if(!ctx) throw std::runtime_error("projection allocation");
        auto * w=ggml_new_tensor_2d(ctx,type,ne[0],ne[1]);
        weights.seekg(gguf_get_data_offset(gguf)+gguf_get_tensor_offset(gguf,id)+expert*ggml_nbytes(w));
        weights.read((char *)w->data,ggml_nbytes(w)); if(!weights) throw std::runtime_error("projection read");
        auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ne[0],1);
        std::ifstream input(std::filesystem::path(manifest).parent_path()/c["input"].get<std::string>(),std::ios::binary);
        input.read((char *)x->data,ggml_nbytes(x));
        if(!input || input.peek()!=EOF) throw std::runtime_error("projection input shape");
        for(int64_t i=0;i<ne[0];i++) if(!std::isfinite(((float *)x->data)[i])) throw std::runtime_error("non-finite input");
        auto * out=ggml_mul_mat(ctx,w,x); auto * graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,out);
        if(ggml_graph_compute_with_ctx(ctx,graph,threads)!=GGML_STATUS_SUCCESS) throw std::runtime_error("projection compute");
        trace.dump("hy.projection."+name+"."+std::to_string(expert),(const float *)out->data,ne[1],json::array({ne[1]}));
        ggml_free(ctx);
    }
    gguf_free(gguf);
}
int main(int argc,char ** argv) try {
    if(argc!=8) throw std::runtime_error("usage: oracle MODE MODEL FIXTURE OUT THREADS MAXTOK INTERLEAVE");
    std::string mode=argv[1],out=argv[4]; int threads=std::stoi(argv[5]),limit=std::stoi(argv[6]); bool interleave=std::stoi(argv[7]);
    std::filesystem::create_directories(out); Trace trace{out,{}};
    if(mode=="projections") { projections(argv[2],argv[3],trace,threads); return 0; }
    llama_backend_init(); auto mp=llama_model_default_params(); mp.n_gpu_layers=0; mp.vocab_only=mode!="translate";
    mp.use_extra_bufts=std::getenv("HY_ORACLE_REPACK")!=nullptr;
    auto * model=llama_model_load_from_file(argv[2],mp); if(!model) throw std::runtime_error("model load failed");
    auto * vocab=llama_model_get_vocab(model);
    if(mode=="tokens") {
        std::ifstream f(argv[3]); json corpus; f>>corpus;
        for(auto & c:corpus) c["ids"]=tokenize(vocab,c["text"],c["add_special"],c["parse_special"]);
        std::ofstream(out+"/tokens.json")<<corpus.dump(2)<<'\n'; llama_model_free(model); return 0;
    }
    std::ifstream pcm(argv[3],std::ios::binary|std::ios::ate); auto bytes=pcm.tellg();
    if(bytes<=0 || bytes%4) throw std::runtime_error("invalid f32 PCM");
    std::vector<float> samples((size_t)bytes/4); pcm.seekg(0); pcm.read((char *)samples.data(),bytes);
    auto ap=mtmd_context_params_default(); ap.use_gpu=false; ap.warmup=false; ap.n_threads=threads;
    ap.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED; ap.cb_eval=callback; ap.cb_eval_user_data=&trace;
    auto * audio=mtmd_init_from_file(argv[2],model,ap); if(!audio) throw std::runtime_error("audio init failed");
    llama_context * ctx=nullptr; int pos=0;
    if(mode=="translate") {
        auto cp=llama_context_default_params(); cp.n_ctx=1024; cp.n_batch=256; cp.n_ubatch=256;
        cp.n_threads=cp.n_threads_batch=threads; cp.type_k=cp.type_v=GGML_TYPE_F32;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED; cp.offload_kqv=false; cp.cb_eval=callback; cp.cb_eval_user_data=&trace;
        ctx=llama_init_from_model(model,cp); if(!ctx) throw std::runtime_error("context init failed");
    }
    auto decode=[&](const std::vector<llama_token> & ids,const float * embd=nullptr,int n=0) {
        if(!ctx) return;
        if(!embd) n=ids.size(); if(n==0) return;
        auto batch=llama_batch_init(n,embd?2048:0,1); batch.n_tokens=n;
        for(int i=0;i<n;i++) {
            if(embd) std::memcpy(batch.embd+i*2048,embd+i*2048,2048*sizeof(float)); else batch.token[i]=ids[i];
            batch.pos[i]=pos++; batch.n_seq_id[i]=1; batch.seq_id[i][0]=0; batch.logits[i]=(i==n-1);
        }
        int err=llama_decode(ctx,batch); llama_batch_free(batch); if(err) throw std::runtime_error("decode failed");
    };
    std::string prefix="<｜hy_begin▁of▁sentence｜><｜reasoning_mode｜>reasoning_effort:no_think<｜hy_User｜>请完成下面的同声传译任务，将陆续收到的语音翻译成中文。请结合上下文边听边译，输出能够确定的译文，不必等待整句结束。";
    std::string suffix="<｜hy_Assistant｜><think></think>"; if(interleave) suffix+="<｜hy_place▁holder▁no▁790｜>";
    decode(tokenize(vocab,prefix));
    json turns=json::array(); int generated=0; bool done=false;
    for(size_t offset=0;offset<samples.size() && !done && generated<limit;) {
        size_t n=std::min<size_t>(offset?12800:25600,samples.size()-offset);
        auto * bitmap=mtmd_bitmap_init_from_audio(n,samples.data()+offset);
        mtmd_input_part part{nullptr,bitmap}; const mtmd_input_part * pp=&part;
        auto * chunks=mtmd_input_chunks_init();
        if(mtmd_tokenize_from_parts(audio,chunks,&pp,1,false)) throw std::runtime_error("audio preprocess failed");
        if(offset) decode(tokenize(vocab,"<｜hy_User｜>"));
        for(size_t i=0;i<mtmd_input_chunks_size(chunks);i++) {
            auto * c=mtmd_input_chunks_get(chunks,i);
            if(mtmd_input_chunk_get_type(c)==MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t nt=0; auto * t=mtmd_input_chunk_get_tokens_text(c,&nt);
                decode(std::vector<llama_token>(t,t+nt));
            } else {
                if(mtmd_encode_chunk(audio,c)) throw std::runtime_error("audio encode failed");
                int nt=mtmd_input_chunk_get_n_tokens(c); auto * data=mtmd_get_output_embd(audio);
                decode({},data,nt);
            }
        }
        decode(tokenize(vocab,suffix)); json tokens=json::array();
        while(ctx && !done && generated<limit) {
            auto * logits=llama_get_logits_ith(ctx,-1); int nv=llama_vocab_n_tokens(vocab);
            trace.dump("hy.logits",logits,nv,json::array({nv}));
            llama_token next=std::max_element(logits,logits+nv)-logits;
            tokens.push_back(next); generated++;
            done=(next==llama_vocab_eos(vocab) || generated==limit);
            if(!done) decode({next});
            if(llama_vocab_is_eog(vocab,next) || done) break;
        }
        turns.push_back(tokens); offset+=n; mtmd_input_chunks_free(chunks); mtmd_bitmap_free(bitmap);
    }
    if(ctx) std::ofstream(out+"/turns.json")<<turns.dump(2)<<'\n';
    mtmd_free(audio); if(ctx) llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
} catch(const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
