from operations.remove import remove_target as rm
from seem.masks import middleware
from paint.crutils import get_crfill_model, process_image_via_crfill, ab8, ab64


def replace_target(opt, old_noun, new_noun, mask_generator=None, label_done=None):
    assert mask_generator != None, 'mask_generator not initialized'
    img = Image.open(opt.in_dir).convert('RGB')
    img = img.resize((ab64(img.size[0]), ab64(img.size[1])))
    res, seem_masks = middleware(opt=opt, image=img, reftxt=opt.edit_txt, tasks=['Text', 'Panopic'])



    img_np, img_dragged, img_obj, img_mask, img_pil, label_done = rm(opt, old_noun, mask_generator, label_done)
    # print(img_dragged.shape, img_obj.shape)

    img_dragged, img_obj = Image.fromarray(np.uint8(img_dragged)), Image.fromarray(np.uint8(img_obj))
    img_dragged_, img_obj_ = Image.fromarray(np.uint8(img_np * img_mask)), Image.fromarray(
        np.uint8(img_np * (1. - img_mask)))
    img_dragged.save('./tmp/test_out/dragged.jpg')
    img_obj.save('./tmp/test_out/obj.jpg')
    img_dragged_.save('./tmp/test_out/dragged_.jpg')
    img_obj_.save('./tmp/test_out/obj_.jpg')

    removed_pil = process_image_via_crfill(img_np, img_mask, opt)  # automatically getting model
    removed_pil.save(f'static-crfill/{opt.out_name}')

    img, mask_all = middleware(opt, removed_pil, old_noun)




