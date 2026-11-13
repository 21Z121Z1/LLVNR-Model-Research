#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unistd.h>

struct PrecisionNode {
    id<MTLComputePipelineState> pipeline;
    std::vector<std::string> inputs;
    std::string name;
    NSUInteger count;
};
struct PrecisionModel {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLBuffer> weights;
    std::unordered_map<std::string,id<MTLBuffer>> buffers;
    std::unordered_map<std::string,NSUInteger> sizes;
    std::vector<PrecisionNode> nodes;
};
static thread_local std::string precisionError;
extern "C" const char *llvnr_precision_error(){return precisionError.c_str();}
extern "C" void *llvnr_precision_create(const char *directory){
    @autoreleasepool {
        auto model=new PrecisionModel();
        NSString *root=[NSString stringWithUTF8String:directory];
        auto fail=[&](NSString *message)->void*{precisionError=message.UTF8String;delete model;return nullptr;};
        model->device=MTLCreateSystemDefaultDevice();model->queue=[model->device newCommandQueue];
        NSError *error=nil;
        NSData *manifest=[NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"prefix.json"]];
        NSDictionary *plan=manifest ? [NSJSONSerialization JSONObjectWithData:manifest options:0 error:&error] : nil;
        if(!plan)return fail(error.localizedDescription ?: @"Missing precision manifest");
        NSString *source=[NSString stringWithContentsOfFile:[root stringByAppendingPathComponent:@"prefix.metal"] encoding:NSUTF8StringEncoding error:&error];
        MTLCompileOptions *options=[MTLCompileOptions new];options.fastMathEnabled=NO;
        id<MTLLibrary> library=source ? [model->device newLibraryWithSource:source options:options error:&error] : nil;
        if(!library)return fail(error.localizedDescription ?: @"Metal compilation failed");
        NSData *weights=[NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"prefix_weights.bin"]];
        model->weights=[model->device newBufferWithBytes:weights.bytes length:MAX(weights.length,2u) options:MTLResourceStorageModeShared];
        for(NSDictionary *node in plan[@"nodes"]){
            NSString *name=node[@"name"];NSUInteger count=1;
            for(NSNumber *dimension in node[@"shape"])count*=dimension.unsignedIntegerValue;
            NSUInteger page=(NSUInteger)getpagesize();
            id<MTLBuffer> buffer=[model->device newBufferWithLength:((count*2+page-1)&~(page-1)) options:MTLResourceStorageModeShared];
            if(!buffer)return fail(@"Metal allocation failed");
            model->buffers[name.UTF8String]=buffer;
            model->sizes[name.UTF8String]=count*2;
            if([node[@"input"] boolValue])continue;
            id<MTLFunction> function=[library newFunctionWithName:node[@"function"]];
            id<MTLComputePipelineState> pipeline=function ? [model->device newComputePipelineStateWithFunction:function error:&error] : nil;
            if(!pipeline)return fail(error.localizedDescription ?: @"Missing precision kernel");
            PrecisionNode record;record.pipeline=pipeline;record.count=count;record.name=name.UTF8String;
            for(NSString *input in node[@"inputs"])record.inputs.push_back(input.UTF8String);
            model->nodes.push_back(record);
        }
        return model;
    }
}
extern "C" void *llvnr_precision_buffer(void *handle,const char *name,size_t *bytes){
    auto model=static_cast<PrecisionModel*>(handle);
    auto found=model->buffers.find(name);
    if(found==model->buffers.end())return nullptr;
    if(bytes)*bytes=model->sizes.at(name);
    return found->second.contents;
}
extern "C" int llvnr_precision_execute(void *handle){
    @autoreleasepool {
        auto model=static_cast<PrecisionModel*>(handle);
        id<MTLCommandBuffer> command=[model->queue commandBuffer];
        if(!command)return -1;
        for(auto &node:model->nodes){
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            if(!encoder)return -1;
            [encoder setComputePipelineState:node.pipeline];
            for(size_t i=0;i<node.inputs.size();i++)[encoder setBuffer:model->buffers.at(node.inputs[i]) offset:0 atIndex:i];
            [encoder setBuffer:model->weights offset:0 atIndex:4];
            [encoder setBuffer:model->buffers.at(node.name) offset:0 atIndex:5];
            NSUInteger threads=MIN((NSUInteger)256,node.pipeline.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(node.count,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
            [encoder endEncoding];
        }
        [command commit];[command waitUntilCompleted];
        if(command.error){precisionError=command.error.localizedDescription.UTF8String;return -1;}
        return 0;
    }
}
extern "C" void llvnr_precision_release(void *handle){delete static_cast<PrecisionModel*>(handle);}
