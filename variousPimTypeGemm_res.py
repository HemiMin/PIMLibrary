combined_res = open('combined_res.txt', 'a')
combined_res.write('\n\n==== PimType GEMM Test ====\n')
combined_res.write('in_h\tin_w\tot_h\tot_w\treorderW\tprepare\trunKernel\tapiCall\tallocH\tallocD\tcpyH2D\tcpyD2H\tdeallocH\tdeallocD\ttotal\n\n')

def extract_time(line):
    return float(line.split(':')[1].strip())

i_h = [1,4,10,20,40]
i_w = [256,512,1024,1536,2048,4096]
o_w = [4096]

for ih in i_h:
    for iw in i_w:
        for ow in o_w:
            res_file = 'PimTypeGemmRes/gemm_' + str(ih) + '_' + str(iw) + '_' + str(ih) + '_' + str(ow)
            res = open(res_file, 'r')

            start_record = False
            reorderW = 0
            prepare = 0
            run_kernel = 0
            api_call = 0
            init = 0
            allocH=0
            allocD=0
            deallocH=0
            deallocD=0
            cpyH2D=0
            cpyD2H=0
            executeGemm=0
            total=0

            error=False

            content = res.read()
            if 'Memory access fault' in content or 'fail' in content or 'Fail' in content:
                error=True

            res.close()
            if not error:
                res = open(res_file, 'r')
                for line in res.readlines():
                    if not start_record:
                        if 'WarmUp' in line:
                            start_record = True
                        else:
                            continue
                    else:
                        if 'ReorderW' in line:
                            reorderW += extract_time(line)
                        elif 'Prepare' in line:
                            prepare += extract_time(line)
                        elif 'RunGemmKernel' in line:
                            run_kernel += extract_time(line)
                        elif 'PimExecuteGemm' in line:
                            api_call += extract_time(line)
                        elif 'Time taken to initialize PIM' in line:
                            init += extract_time(line)
                        elif 'to allocH PIM' in line:
                            allocH += extract_time(line)
                        elif 'to allocD PIM' in line:
                            allocD += extract_time(line)
                        elif 'deallocH PIM' in line:
                            deallocH += extract_time(line)
                        elif 'deallocD PIM' in line:
                            deallocD += extract_time(line)
                        elif 'copyH2D_time_ PIM' in line:
                            cpyH2D += extract_time(line)
                        elif 'copyD2H_time_ PIM' in line:
                            cpyD2H += extract_time(line)
                        elif 'to pim execute operation' in line:
                            executeGemm += extract_time(line)
                        elif 'to execute operation' in line:
                            total += extract_time(line)

                reorderW = float(reorderW/5)
                prepare = float(prepare/5)
                run_kernel = float(run_kernel/5)
                api_call = float(api_call/5)

                combined_res.write(f'{str(ih)}\t{str(iw)}\t{str(ih)}\t{str(ow)}\t{round(reorderW,3)}\t{round(prepare,3)}\t{round(run_kernel,3)}\t{round(api_call,3)}\t{round(allocH,3)}\t{round(allocD,3)}\t{round(cpyH2D,3)}\t{round(cpyD2H,3)}\t{round(deallocH,3)}\t{round(deallocD,3)}\t{round(total,3)}\n')
                res.close()
            else:
                combined_res.write(f'{str(ih)}\t{str(iw)}\t{str(ih)}\t{str(ow)}\tError\n')
combined_res.close();
