'''
Single_command_exp
    - This file is used to run the single command experiment
    - Using COCO_val (2017) as the dataset (5k images)
    - Using the following commands:
        - add
        - remove
        - replace
        - locate        
'''
from single_task import *
from task_planning import *
import os
import json

def single_command(opt, img, modify_txt, test_type='add'):
    '''
    doing one single command to the image
    img:        (string)    (e.g.) 000000000139.jpg \\
    modify_txt: (string)    (e.g.) "add a person" \\ file naem: 000000000139.txt
    '''
    
    # for test, set "preload_all_models" = True is highly recommanded
    api_key = opt.api_key
    net_proxy = opt.net_proxy
    engine = opt.engine
    print(f'Using: {engine}')

    assert os.path.exists(opt.in_dir), f'File Not Exists: {opt.in_dir}'
    opt.device = "cuda" if torch.cuda.is_available() else "cpu"
    """
    base_dir:
        -> semantic (must been done)
        -> remove (if done)
        -> replace (if done)
        -> locate (if done)
        -> add (if done)
    """

    sorted_class = test_type
    # reset opt
    opt.in_dir = f"../autodl-tmp/COCO/test2017/{img}"
    # edit_txt is the string in modify_txt
    opt.edit_txt = modify_txt
    print(f'current class: <{sorted_class}>')
    preloaded_model = preload_all_models(opt)
    preloaded_agent = preload_all_agents(opt)

    if 'remove' in sorted_class:
        # find the target -> remove -> recover the scenery
        agent = preloaded_agent['find target to be removed']
        target_noun = get_response(agent, opt.edit_txt)
        print(f'\'{target_noun}\' will be removed')
        output_image, *_ = Remove_Me_lama(
                                opt, target_noun, input_pil = None,
                                dilate_kernel_size=opt.dilate_kernel_size,
                                preloaded_model = preloaded_model
                            )
        output_image.save(f'../autodl-tmp/COCO/outputs/remove/{img}')
        # TODO: recover the scenery for img_dragged in mask

    elif 'replace' in sorted_class:
        # find the target -> remove -> recover the scenery -> add the new
        replace_agent = preloaded_agent['find target to be replaced']
        replace_tuple = get_response(replace_agent, opt.edit_txt)
        print(f'replace_tuple = {replace_tuple}')
        old_noun, new_noun = get_replace_tuple(replace_tuple)
        print(f'Replacement will happen: \'{old_noun}\' -> \'{new_noun}\'')

        # TODO: replace has no need of an agent; original mask and box is necessary!
        rescale_agent = preloaded_agent['rescale bbox for me']
        diffusion_agent = preloaded_agent['expand diffusion prompts for me']
        output_image = replace_target(opt, old_noun, new_noun, edit_agent=rescale_agent, expand_agent=diffusion_agent, preloaded_model = preloaded_model)
        output_image.save(f'../autodl-tmp/COCO/outputs/replace/{img}')

    elif 'move' in sorted_class:
        # find the (move-target, move-destiny) -> remove -> recover the scenery -> paste the origin object
        move_agent = preloaded_agent['arrange a new bbox for me']
        noun_agent = preloaded_agent['find target to be moved']
        target_noun = get_response(noun_agent, opt.edit_txt)
        print(f'target_noun: {target_noun}')
        output_image = create_location(opt, target_noun, edit_agent=move_agent, preloaded_model = preloaded_model)
        output_image.save(f'../autodl-tmp/COCO/outputs/locate/{img}')

    elif 'add' in sorted_class:
        add_agent = preloaded_agent['find target to be added']
        ans = get_response(add_agent, opt.edit_txt)
        print(f'tuple_ans: {ans}')
        name, num, place = get_add_tuple(ans)
        print(f'name = {name}, num = {num}, place = {place}')

        arrange_agent = preloaded_agent['generate a new bbox for me'] if '<NULL>' in place else preloaded_agent['adjust bbox for me']
        diffusion_agent = preloaded_agent['expand diffusion prompts for me']
        output_image = Add_Object(opt, name, num, place, edit_agent=arrange_agent, expand_agent=diffusion_agent, preloaded_model = preloaded_model)
        output_image.save(f'../autodl-tmp/COCO/outputs/add/{img}')



def single_command_exp(opt):
    # Checking COCO_val (2017) dataset
    if os.path.exists('autodl-pub/COCO2017'):
        print('COCO_val (2017) dataset found')

    # saving captions_val2017.json to a variable
    # captions_val2017 = json.load(open('../autodl-pub/COCO2017/annotations/captions_val2017.json'))
    # iterate every image in the COCO_val (2017) dataset
    for img in os.listdir('../autodl-pub/COCO2017/val2017'):
        # deleting front 0s to get the image id
        img_id = img.split('.')[0].lstrip('0')
        # getting the caption of the image
        caption = captions_val2017['annotations'][int(img_id)]['caption']
        # generate modifying command & testing command in modify.txt and test.txt
        # get modify_txt(a string) from autodl-tmp/COCO/prompt/add/000000000139.txt
        # use the modify.txt to run the single command experiment
        for test_type in ['add', 'remove', 'move', 'replace']:
            with open(f"../autodl-tmp/COCO/prompt/{test_type}/{img.split('.')[0]}.txt") as f:
                modify_txt = f.read()
            single_command(opt, img, modify_txt, test_type)

        # use the test.txt to test the output image
        # (e.g.) test_modified_img(img, output_img, test_txt)


if __name__ == "__main__":
    opt = get_arguments()
    single_command_exp(opt)
