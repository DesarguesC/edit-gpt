## add mask control inside inst-inpaint ?

## remove (done)

## replce

1. preprocess: generate [mask, box] list using original image

2. remove: remove target noun gained from GPT-3.5. (DONE)

3. [mask, box] + target noun (replace), generate box location via GPT-3.5

4. generate new object, fine up/down-sampling scale using box predicted by SEEM and campare it to the box gained on the previous step

5. add the object to the image via Paint-by-Example

## move

1. preprocess: ... (pay attention to target location)

2. remove: ...

3. [mask, box] + target location (move to), generate box location via GPT-3.5 (pay attention to the object size)

4. calculate up/downsampling scale: edit-prompt + GPT-3.5, resize the box

5. add the object to the image via Paint-by-Example
