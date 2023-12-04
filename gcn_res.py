combined_res = open('combined_res.txt', 'a')
combined_res.write('\n\n==== GCN ====\n')

def extract_time(line):
    return float(line.split(':')[1].strip())

res_file = 'gcn_res'
res = open(res_file, 'r')
idx=[0,1,2,3]

start_record = False
reorderW = [0,0,0,0]
prepare = [0,0,0,0]
run_kernel = [0,0,0,0]
api_call = [0,0,0,0]
align = [0] * 11
init = 0
allocH=0
allocD=0
deallocH=0
deallocD=0
cpyH2D=0
cpyD2H=0
cpyaligned=0
executeGemm=0
total=0

error=False

content = res.read()
if 'Memory access fault' in content or 'fail' in content or 'Fail' in content:
    error=True

res.close()
if not error:
    res = open(res_file, 'r')
    i = 0
    for line in res.readlines():
        if not start_record:
            if 'WarmUp' in line:
                start_record = True
            else:
                continue
        else:
            if 'ReorderW' in line:
                reorderW[i] += extract_time(line)
            elif 'Prepare' in line:
                prepare[i] += extract_time(line)
            elif 'RunGemmKernel' in line:
                run_kernel[i] += extract_time(line)
            elif 'ExecuteGemm time (us)' in line:
                api_call[i] += extract_time(line)
                if i==3:
                    i = 0
                else:
                    i += 1
            elif 'PimAlignInput' in line:
                align[0] += extract_time(line)
            elif 'PimAlignAdj' in line:
                align[1] += extract_time(line)
            elif 'PimAlignL1W' in line:
                align[2] += extract_time(line)
            elif 'PimAlignL2W' in line:
                align[3] += extract_time(line)
            elif 'PimAlignL1V1 time' in line:
                align[4] += extract_time(line)
            elif 'PimAlignL1V2 time' in line:
                align[5] += extract_time(line)
            elif 'PimAlignL1V1_2' in line:
                align[6] += extract_time(line)
            elif 'PimAlignL1V2_2' in line:
                align[7] += extract_time(line)
            elif 'PimAlignL2V1 time' in line:
                align[8] += extract_time(line)
            elif 'PimAlignOut' in line:
                align[9] += extract_time(line)
            elif 'PimAlignL2V1_2' in line:
                align[10] += extract_time(line)
            elif 'PimCopyAlignedData' in line:
                cpyaligned += extract_time(line)
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

    reorderW = [float(a/5) for a in reorderW]
    prepare = [float(a/5) for a in prepare]
    run_kernel = [float(a/5) for a in run_kernel]
    api_call = [float(a/5) for a in api_call]
    align = [float(a/5) for a in align]
    cpyaligned = float(cpyaligned/5)
   # prepare = float(prepare/5)
   # run_kernel = float(run_kernel/5)
   # api_call = float(api_call/5)

combined_res.write('reorderW\tprepare\trunKernel\tapiCall\n')
for i in range(4):
    combined_res.write(f'{round(reorderW[i],3)}\t{round(prepare[i],3)}\t{round(run_kernel[i],3)}\t{round(api_call[i],3)}\n')
combined_res.write('\nalign\n')
for i in range(11):
    combined_res.write(f'{round(align[i],3)}\n')
combined_res.write('\ncopyAlignedData\tallocH\tallocD\tcpyH2D\tcpyD2H\tdeallocH\tdeallocD\texecuteGemm\ttotal\n')
combined_res.write(f'{round(cpyaligned,3)}\t{round(allocH,3)}\t{round(allocD,3)}\t{round(cpyH2D,3)}\t{round(cpyD2H,3)}\t{round(deallocH,3)}\t{round(deallocD,3)}\t{round(executeGemm,3)}\t{round(total,3)}\n')
combined_res.close()
