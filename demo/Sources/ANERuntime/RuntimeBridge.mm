#import <Foundation/Foundation.h>
#include "include/ANERuntime.h"
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>
#include <cstdlib>
#include <sys/sysctl.h>

extern "C" {
struct ane_e5rt_program;
ane_e5rt_program *ane_e5rt_program_compile(const char *, const char *, uint64_t,
    const char *const *, const size_t *, size_t, const char *const *, const size_t *, size_t);
int ane_e5rt_program_input_buffer(ane_e5rt_program *, const char *, void **, size_t *);
int ane_e5rt_program_output_buffer(ane_e5rt_program *, const char *, void **, size_t *);
int ane_e5rt_program_execute(ane_e5rt_program *);
void ane_e5rt_program_release(ane_e5rt_program *);
void *llvnr_precision_create(const char *);
void *llvnr_precision_buffer(void *,const char *,size_t *);
int llvnr_precision_execute(void *);
void llvnr_precision_release(void *);
}

struct BufferBinding {void *pointer;size_t bytes;};
struct StateCopy {void *source;void *destination;size_t bytes;};
struct ModelHandle {
    ane_e5rt_program *ane=nullptr;
    void *precision=nullptr;
    std::unordered_map<std::string,BufferBinding> inputs,outputs;
    std::vector<StateCopy> copies;
    ~ModelHandle(){if(ane)ane_e5rt_program_release(ane);if(precision)llvnr_precision_release(precision);}
};

void *llvnr_compile(const char *model_directory, const char *cache_directory) {
    @autoreleasepool {
        NSString *directory = [NSString stringWithUTF8String:model_directory];
        NSData *data = [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"manifest.json"]];
        NSDictionary *manifest = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
        if (!manifest) return nullptr;
        std::vector<const char *> inputs, outputs;
        std::vector<size_t> input_sizes, output_sizes;
        NSArray *aneInputs=manifest[@"ane_inputs"] ?: manifest[@"inputs"];
        NSArray *aneOutputs=manifest[@"ane_outputs"] ?: manifest[@"outputs"];
        for (NSDictionary *p in aneInputs) {
            inputs.push_back([p[@"port"] UTF8String]); input_sizes.push_back([p[@"bytes"] unsignedLongLongValue]);
        }
        for (NSDictionary *p in aneOutputs) {
            outputs.push_back([p[@"port"] UTF8String]); output_sizes.push_back([p[@"bytes"] unsignedLongLongValue]);
        }
        auto model=new ModelHandle();
        model->ane=ane_e5rt_program_compile([[directory stringByAppendingPathComponent:@"model.mil"] UTF8String],
            cache_directory, 4, inputs.data(), input_sizes.data(), inputs.size(), outputs.data(), output_sizes.data(), outputs.size());
        if(!model->ane){delete model;return nullptr;}
        if([manifest[@"execution_profile"] isEqual:@"metal_state_ane_denoiser"]){
            model->precision=llvnr_precision_create([[directory stringByAppendingPathComponent:@"precision"] UTF8String]);
            if(!model->precision){delete model;return nullptr;}
            for(NSDictionary *port in aneInputs){
                size_t sourceSize=0,destinationSize=0;void *destination=nullptr;
                void *source=llvnr_precision_buffer(model->precision,[port[@"name"] UTF8String],&sourceSize);
                if(source){
                    if(ane_e5rt_program_input_buffer(model->ane,[port[@"port"] UTF8String],&destination,&destinationSize) || sourceSize!=destinationSize){delete model;return nullptr;}
                    model->copies.push_back({source,destination,sourceSize});
                }
            }
        }
        for(int kind=0;kind<2;kind++){
            for(NSDictionary *port in manifest[kind ? @"outputs" : @"inputs"]){
                size_t bytes=0;void *pointer=model->precision ? llvnr_precision_buffer(model->precision,[port[@"name"] UTF8String],&bytes) : nullptr;
                if(!pointer){
                    NSArray *ports=kind ? aneOutputs : aneInputs;
                    NSDictionary *matched=nil;
                    for(NSDictionary *candidate in ports)if([candidate[@"name"] isEqual:port[@"name"]]){matched=candidate;break;}
                    if(matched){
                        int result=kind ? ane_e5rt_program_output_buffer(model->ane,[matched[@"port"] UTF8String],&pointer,&bytes)
                                        : ane_e5rt_program_input_buffer(model->ane,[matched[@"port"] UTF8String],&pointer,&bytes);
                        if(result)pointer=nullptr;
                    }
                }
                if(!pointer || bytes!=[port[@"bytes"] unsignedLongLongValue]){delete model;return nullptr;}
                (kind ? model->outputs : model->inputs)[[port[@"port"] UTF8String]]={pointer,bytes};
            }
        }
        return model;
    }
}
int llvnr_host_ane_family(void) {
    size_t size = 0;
    if (sysctlbyname("machdep.cpu.brand_string", nullptr, &size, nullptr, 0) != 0 || size == 0) return -1;
    char *brand = static_cast<char *>(std::calloc(size + 1, 1));
    if (!brand) return -1;
    const int result = sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0);
    if (result != 0) { std::free(brand); return -1; }
    const char *marker = std::strstr(brand, "Apple M");
    const int generation = marker ? std::atoi(marker + 7) : 0;
    std::free(brand);
    switch (generation) {
        case 1: return 2; // M1 / H13
        case 2: return 3; // M2 / H14
        case 3: return 4; // M3 / H15
        case 4: return 5; // M4 / A16 tier
        case 5: return 5; // M5 / A16 tier
        default: return -1;
    }
}

void *llvnr_input(void *m, const char *p, size_t *b) {
    auto model=static_cast<ModelHandle*>(m);auto found=model->inputs.find(p);
    if(found==model->inputs.end())return nullptr;
    if(b)*b=found->second.bytes;return found->second.pointer;
}
void *llvnr_output(void *m, const char *p, size_t *b) {
    auto model=static_cast<ModelHandle*>(m);auto found=model->outputs.find(p);
    if(found==model->outputs.end())return nullptr;
    if(b)*b=found->second.bytes;return found->second.pointer;
}
int llvnr_execute(void *m) {
    auto model=static_cast<ModelHandle*>(m);
    if(model->precision){
        if(llvnr_precision_execute(model->precision))return -1;
        for(const auto &copy:model->copies)std::memcpy(copy.destination,copy.source,copy.bytes);
    }
    return ane_e5rt_program_execute(model->ane);
}
void llvnr_release(void *m) { delete static_cast<ModelHandle*>(m); }
int llvnr_acquire_lock(void) {
    int fd = open("/tmp/llvnr_ane.lock",O_RDWR|O_CREAT,0600);
    if (fd < 0) return -1;
    struct flock lock = {}; lock.l_type=F_WRLCK; lock.l_whence=SEEK_SET;
    if (fcntl(fd,F_SETLK,&lock)<0) {close(fd); return -1;}
    return fd;
}
void llvnr_release_lock(int fd) { if(fd>=0) close(fd); }
