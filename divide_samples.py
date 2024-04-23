import os
import uproot
import awkward as ak
import pyarrow.parquet as pq

def splitfile(fn,nsplit,output):
  arr = ak.from_parquet(fn)
  nevt = int(len(arr)/nsplit)
  idx = 0
  iiter = 0
  while idx < len(arr):
    arr_out = arr[idx:idx+nevt]
    ak.to_parquet(arr_out, output+'_{}.parquet'.format(iiter), compression='LZ4', compression_level=4)
    idx = idx+nevt
    iiter += 1
  return


def makefile(fn,output):
  f = uproot.open(fn)
  f = f['MLTree/tree']
  arr = f.arrays(f.keys(),library='ak')
  nevt = len(arr)
  print(fn,len(arr))
  arr_train = arr[:int(nevt*0.7)]
  arr_test = arr[int(nevt*0.7):]
  ak.to_parquet(arr_train, output+'_train.parquet', compression='LZ4', compression_level=4)
  ak.to_parquet(arr_test, output+'_test.parquet', compression='LZ4', compression_level=4)

def main():
  dirs = '/scratch-cbe/users/ang.li/SoftDV/MLTree_MLTree_v1/'
  for s in os.listdir(dirs):
    fn = dirs+s+'/fs/'+s+'.root'
    makefile(fn,s)

#def main():
#  fn_temp = "zjetstonunuht%04i_2018_test.parquet"
#  for m in [200,400,600,800]:
#    splitfile(fn_temp % m, 20, "zjetstonunuht%04i_2018_test" % m)
#  for m in [1200,2500]:
#    splitfile(fn_temp % m, 2, "zjetstonunuht%04i_2018_test" % m)


if __name__ == '__main__':
  main()
