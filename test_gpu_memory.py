from preload_utils import *
import torch, os, time
from prompt.arguments import get_arguments

def test(name):
    print(name)
    os.system('nvidia-smi')
    os.system(f'echo "{name.upper()}" >> memory_test.txt && nvidia-smi >> memory_test.txt')
    print()

def main():
    opt = get_arguments()
    if os.path.isfile('./memory_test.txt'):
        os.system('rm memory_test.txt')
    
    start_time = time.time()

    ip2p = preload_ip2p(opt)
    test('ip2p')
    del ip2p

    xl = preload_XL_generator(opt)
    test('xl')
    del xl

    xl_ad = preload_XL_adapter_generator(opt)
    test('xl_ad')
    del xl_ad

    v1_5 = preload_v1_5_generator(opt)
    test('v1.5')
    del v1_5

    paint_by_example = preload_paint_by_example_model(opt)
    test('paint_by_example')
    del paint_by_example

    sam = preload_sam_generator(opt)
    test('sam')
    del sam

    seem = preload_seem_detector(opt)
    test('seem')
    del seem

    lama = preload_lama_remover(opt)
    test('lama')
    del lama

    end_time = time.time()

    print(f'time cost: {end_time - start_time}')

if __name__ == '__main__':
    main()


