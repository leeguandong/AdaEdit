#!/bin/bash
# AdaEdit Example Scripts

# Example 1: Object Replacement with Progressive Sigmoid Schedule
python adaedit.py \
    -i examples/cat.jpg \
    -sp "A photo of a cat on a sofa" \
    -tp "A photo of a dog on a sofa" \
    --edit_object "cat" \
    --inject_schedule sigmoid \
    --use_channel_ls \
    --seed 42

# Example 2: Style Transfer (no Latents-Shift for style edits)
python adaedit.py \
    -i examples/portrait.jpg \
    -sp "A portrait photo" \
    -tp "A portrait in anime style" \
    --edit_type style \
    --inject_schedule cosine

# Example 3: Attribute Change with Cosine Schedule
python adaedit.py \
    -i examples/scene.jpg \
    -sp "A street scene in daytime" \
    -tp "A street scene at night" \
    --inject_schedule cosine \
    --use_channel_ls \
    --channel_ls_temp 1.0

# Example 4: Low VRAM Mode
python adaedit.py \
    -i examples/image.jpg \
    -sp "A photo of a person" \
    -tp "A photo of a person smiling" \
    --offload

# Example 5: Comparison of Different Schedules
for schedule in binary sigmoid cosine linear; do
    python adaedit.py \
        -i examples/test.jpg \
        -sp "A photo of a car" \
        -tp "A photo of a red car" \
        --inject_schedule $schedule \
        --seed 42
done
