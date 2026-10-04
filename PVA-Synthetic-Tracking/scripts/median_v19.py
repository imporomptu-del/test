"""Generate a bounded median-selection DAG; no source/media IO on import."""
OLD = '''    // Fixed odd-even sorting network: comparisons select existing float bits.
    #pragma unroll
    for(int p=0;p<25;p++) {
        #pragma unroll
        for(int j=(p&1);j<24;j+=2) {
            float lo=fminf(a[j],a[j+1]), hi=fmaxf(a[j],a[j+1]);
            a[j]=lo; a[j+1]=hi;
        }
    }
'''
SOURCE_SHA = {
 'phase20_cuda_resident.cu':'cf64594508ef47c599e30c6c41fa2c0ee2888c3afa2cdbf37df783113dc8261a',
 'phase20_cuda_median.cu':'452e1cfe96636cb8dad74dd74471507a1318992ec1c62fa43767e2876ebf05f7',
 'phase20_cuda_warp_exact.cu':'7eb54309d9d0e504ec8d0fd86532346d4549a0ce2d7d2b1d8e0943bda45c8374',
 'phase20_cuda_integrated.cu':'ea607056117051b9655c28ecd2ee7bf444fc53f344e629a732a1c51cd88a8366'}
REFERENCE_LIBRARY_SHA='0877b81e2332c329c3bbae2d07a0db5b615c821aa11943c6a68c797f37bdc197'


def network():
    comparators=[]
    def merge(lo,n,step):
        double=step*2
        if double<n:
            merge(lo,n,double); merge(lo+step,n,double)
            for i in range(lo+step,lo+n-step,double):
                comparators.append((i,i+step))
        else:
            comparators.append((lo,lo+step))
    def sort(lo,n):
        if n>1:
            sort(lo,n//2);sort(lo+n//2,n//2);merge(lo,n,1)
    sort(0,32)
    wires=list(range(25))+[None]*7
    nodes={}
    for i,j in comparators:
        a,b=wires[i],wires[j]
        if a is None or b is None:
            wires[i],wires[j]=(b if a is None else a),None
        else:
            k=25+len(nodes);nodes[k]=('min',a,b);nodes[k+1]=('max',a,b)
            wires[i],wires[j]=k,k+1
    final=wires[12]
    assert final is not None
    needed=set()
    def visit(k):
        if k<25 or k in needed:return
        needed.add(k)
        _,a,b=nodes[k];visit(a);visit(b)
    visit(final)
    return [(k,*nodes[k]) for k in sorted(needed)],final


def emit(kind):
    nodes,final=network()
    name=lambda k:f'a[{k}]' if k<25 else f'v{k}'
    lines=[]
    for k,op,a,b in nodes:
        expression=(f'{name(a)} {"&" if op=="min" else "|"} {name(b)}' if kind=='bits'
                    else f'f{op}f({name(a)}, {name(b)})')
        lines.append(f'    {"uint64_t" if kind=="bits" else "float"} v{k} = {expression};')
    return '\n'.join(lines),name(final)


def transform(source):
    if source.count(OLD)!=1:
        raise ValueError('Exactly one frozen median network required')
    code,result=emit('float')
    guard='''    bool special=false;
    #pragma unroll
    for(int i=0;i<25;++i) {
        unsigned int bits=__float_as_uint(a[i]), mag=bits&0x7fffffffU;
        special |= mag>=0x7f800000U || bits==0x80000000U || (mag && mag<0x00800000U);
    }
    if(special) {
'''
    replacement=guard+OLD+'    } else {\n'+code+'\n    a[12] = '+result+';\n    }\n'
    return source.replace(OLD,replacement)


def proof_source():
    code,result=emit('bits')
    return '''#include <cstdint>
#include <cstdio>
int main() {
    uint64_t expected[20]={};
    for(int high=0;high<20;++high)for(int low=0;low<64;++low)
        if(high+__builtin_popcount(unsigned(low))>=13)expected[high]|=uint64_t(1)<<low;
    for(uint32_t block=0;block<(1U<<19);++block) {
        uint64_t a[25];
        a[0]=0xaaaaaaaaaaaaaaaaULL;a[1]=0xccccccccccccccccULL;
        a[2]=0xf0f0f0f0f0f0f0f0ULL;a[3]=0xff00ff00ff00ff00ULL;
        a[4]=0xffff0000ffff0000ULL;a[5]=0xffffffff00000000ULL;
        for(int k=6;k<25;++k)a[k]=((block>>(k-6))&1)?~uint64_t(0):uint64_t(0);
'''+code+'''
        if('''+result+''' != expected[__builtin_popcount(block)]) {
            std::fprintf(stderr,"Mismatch at block %u\\n",block);return 1;
        }
    }
    std::puts("33554432 zero-one cases exact");return 0;
}
'''
