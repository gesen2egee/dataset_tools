import re
import math
from typing import List, Dict, Optional, Tuple, Set
import glob
import numpy as np
from tqdm import tqdm
import argparse
import shutil
import subprocess
import os
import base64
import random
from io import BytesIO
from datetime import datetime, timedelta
import fnmatch
import torch

# 自動安裝所需庫
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.run(["pip", "install", package], check=True)
        __import__(package)

libraries = [
    'tqdm',
    'PIL',
    'pathlib',
    'datetime',
    'matplotlib',
    'natsort',
    'pandas'
]


for lib in libraries:
    install_and_import(lib)
    
from PIL import Image
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from natsort import natsorted
import pandas as pd


try:
    import cv2
except ImportError:    
    subprocess.run(["pip", "install", "Opencv-python"], check=True)
    import cv2


try:
    from imgutils.tagging import get_wd14_tags, tags_to_text, drop_blacklisted_tags, drop_basic_character_tags, drop_overlap_tags
except ImportError:
    print("正在安装 dghs-imgutils[gpu]...")
    subprocess.run(["pip", "install", "dghs-imgutils[gpu]", "--upgrade"], check=True)
    from imgutils.tagging import get_wd14_tags, tags_to_text, drop_blacklisted_tags, drop_basic_character_tags, drop_overlap_tags

try:
    from clip_interrogator import Config, Interrogator, LabelTable, load_list
except ImportError:
    subprocess.run(["pip", "install", "clip-interrogator==0.6.0"], check=True)
    from clip_interrogator import Config, Interrogator, LabelTable, load_list

# 常量
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

# 內建的外表標籤名單image_base_names
appearance_tags = {
    'long hair', 'breasts', 'short hair', 'blue eyes', 'large breasts', 'blonde hair', 'brown hair', 'black hair', 'hair ornament', 'red eyes', 'hat', 'bow', 'animal ears', 'ribbon', 'hair between eyes', 'jewelry', 'very long hair', 'twintails', 'medium breasts', 'brown eyes', 'green eyes', 'blue hair', 'purple eyes', 'tail', 'yellow eyes', 'white hair', 'pink hair', 'grey hair', 'ahoge', 'braid', 'hair ribbon', 'purple hair', 'ponytail', 'multicolored hair', 'sidelocks', 'hair bow', 'earrings', 'red hair', 'small breasts', 'frills', 'hairband', 'horns', 'wings', 'green hair', 'choker', 'glasses', 'pointy ears', 'virtual youtuber', 'hairclip', 'medium hair', 'fang', 'bowtie', 'hood', 'dark skin', 'cat ears', 'blunt bangs', 'hair flower', 'pink eyes', 'necklace', 'hair bun', 'mole', 'hair over one eye', 'rabbit ears', 'orange hair', 'scarf', 'black eyes', 'two-tone hair', 'streaked hair', 'huge breasts', 'halo', 'flat chest', 'twin braids', 'side ponytail', 'animal ear fluff', 'red ribbon', 'aqua eyes', 'dark-skinned female', 'parted bangs', 'two side up', 'v-shaped eyebrows', 'grey eyes', 'orange eyes', 'cat tail', 'symbol-shaped pupils', 'eyelashes', 'loli', 'black headwear', 'mole under eye', 'fox ears', 'no humans', 'fake animal ears', 'muscular', 'single braid', 'gradient hair', 'black choker', 'double bun', 'makeup', 'floating hair', 'aqua hair', 'colored skin', 'swept bangs', 'facial hair', 'heterochromia', 'white headwear', 'alternate hairstyle', 'blue bow', 'fox tail', 'witch hat', 'low twintails', 'one side up', 'scar', 'horse ears', 'wavy hair', 'fangs', 'thick thighs', 'hair intakes', 'blush stickers', 'facial mark', 'thick eyebrows', 'abs', 'horse girl', 'headgear', 'muscular male', 'heart-shaped pupils', 'bob cut', 'drill hair', 'sunglasses', 'pectorals', 'dark-skinned male', 'light brown hair', 'wolf ears', 'black hairband', 'eyepatch', 'scrunchie', 'white bow', 'demon girl', 'cat girl', 'mob cap', 'magical girl', 'eyes visible through hair', 'demon horns', 'sharp teeth', 'single hair bun', 'high ponytail', 'helmet', 'feathers', 'x hair ornament', 'fox girl', 'blue ribbon', 'antenna hair', 'cross', 'hat ribbon', 'crown', 'pink bow', 'robot', 'spiked hair', 'bat wings', 'ear piercing', 'slit pupils', 'bright pupils', 'elf', 'monster girl', 'rabbit tail', 'head wings', 'gem', 'tan', 'red bowtie', 'furry', 'short twintails', 'messy hair', 'lipstick', 'goggles', 'horse tail', 'otoko no ko', 'straight hair', 'feathered wings', 'multiple tails', 'extra ears', 'eyewear on head', 'demon tail', 'dog ears', 'hooded jacket', 'pale skin', 'child', 'red headwear', 'white ribbon', 'ribbon trim', 'colored inner hair', 'hair over shoulder', 'skin fang', 'genderswap', 'mole under mouth', 'side braid', 'third eye', 'scar on face', 'large pectorals', 'baseball cap', 'beard', 'blue headwear', 'claws', 'red neckerchief', 'glowing eyes', 'white pupils', 'short hair with long locks', 'semi-rimless eyewear', 'low ponytail', 'hair flaps', 'twin drills', 'androgynous', 'yellow bow', 'forehead', 'hood up', 'wolf tail', 'neck bell', 'eyeshadow', 'french braid', 'tokin hat', 'curvy', 'faceless', 'crossed bangs', 'colored sclera', 'black wings', 'alternate breast size', 'green bow', 'single horn', 'dragon horns', 'hair scrunchie', 'genderswap \(mtf\)', 'santa hat', 'wide hips', 'pink ribbon', 'half updo', 'freckles', 'demon wings', 'single earring', 'low-tied long hair', 'headset', 'white skin', 'hair rings', 'beads', 'mature male', 'unworn headwear', 'long fingernails', 'toned', 'mole on breast', 'black-framed eyewear', 'short ponytail', 'purple bow', 'round eyewear', 'angel wings', 'one eye covered', 'goggles on head', 'black bowtie', 'braided ponytail', 'red-framed eyewear', 'curly hair', 'empty eyes', 'dot nose', 'futanari', 'kemonomimi mode', 'tanlines', 'hat ornament', 'dragon girl', 'animal hands', 'sideburns', 'jitome', 'faceless male', 'red scarf', 'abyssal ship', 'asymmetrical hair', 'dog tail', 'mini person', 'yellow ribbon', 'top hat', 'sun hat', 'plump', 'furry female', 'white hairband', 'asymmetrical bangs', 'fake tail', 'star hair ornament', 'under-rim eyewear', 'white wings', 'mature female', 'circlet', 'hair down', 'multicolored eyes', 'official alternate hairstyle', 'stubble', 'colored eyelashes', 'crystal', 'hairpin', 'rabbit girl', 'hoop earrings', 'unworn hat', 'blue bowtie', 'long legs', 'tentacle hair', 'eyebrows hidden by hair', 'minigirl', 'green headwear', 'wolf girl', 'light blue hair', 'mini hat', 'military hat', 'brown headwear', 'bespectacled', 'dragon tail', 'striped bow', 'tress ribbon', 'pink lips', 'animal hood', 'short eyebrows', 'scar across eye', 'mustache', 'folded ponytail', 'dog girl', 'furry male', 'forehead mark', 'blue skin', 'heart hair ornament', 'sharp fingernails', 'muscular female', 'red hairband', 'hime cut', 'mouse ears', 'hair bell', 'bandaid on face', 'nurse cap', 'purple ribbon', 'butterfly hair ornament', 'hat flower', 'hair tie', 'straw hat', 'green ribbon', 'visor cap', 'orange bow', 'stud earrings', 'blindfold', 'bags under eyes', 'black sclera', 'low wings', 'antennae', 'headpiece', 'long bangs', 'gag', 'eyeliner', 'beanie', 'red lips', 'white scarf', 'fake horns', 'no nose', 'alternate hair length', 'crown braid', 'tail ornament', 'eighth note', 'sailor hat', 'android', 'round teeth', 'angel', 'hair behind ear', 'cabbie hat', 'flipped hair', 'single side bun', 'bun cover', 'anchor symbol', 'absurdly long hair', 'frog hair ornament', 'arm tattoo', 'on head', 'fairy wings', 'star-shaped pupils', 'bird wings', 'hair over eyes', 'cow print', 'cow ears', 'antlers', 'food-themed hair ornament', 'pink headwear', 'black horns', 'scales', 'hair stick', 'headdress', 'petite', 'feather hair ornament', 'tinted eyewear', 'ringed eyes', 'huge ass', 'mask on head', 'fairy', 'horn ornament', 'cow horns', 'creature', 'mini crown', 'very short hair', 'white fur', 'blue hairband', 'green skin', 'red choker', 'blue halo', 'tiger ears', 'undercut', 'symbol in eye', 'wet hair', 'earmuffs', 'purple headwear', 'alternate eye color', 'flat cap', 'triangular headpiece', 'snake hair ornament', 'star print', 'cone hair bun', 'paw print', 'curled horns', 'ice wings', 'belly', 'bald', 'whisker markings', 'mechanical halo', 'red horns', 'animal hat', 'alternate hair color', 'mecha musume', 'mechanical arms', 'raccoon ears', 'bishounen', 'scar on cheek', 'pink halo', 'white choker', 'unworn eyewear', 'shoulder tattoo', 'lolita hairband', 'vampire', 'facepaint', 'animal nose', 'hair up', 'long sideburns', 'star earrings', 'crescent hair ornament', 'idol', 'mouse tail', 'ribbon choker', 'tate eboshi', 'hat feather', 'garrison cap', 'white eyes', 'head wreath', 'deep skin', 'frilled bow', 'giant', 'tilted headwear', 'animal on head', 'blunt ends', 'grey skin', 'ear ornament', 'asymmetrical wings', 'two tails', 'forehead jewel', 'stitches', 'facial tattoo', 'headphones around neck', 'cross necklace', 'crescent hat ornament', 'toned male', 'no pupils', 'glowing eye', 'old', 'mermaid', 'fish tail', 'constricted pupils', 'split-color hair', 'gyaru', 'head fins', 'leaf hair ornament', 'rabbit hair ornament', 'spiked collar', 'draph', 'neck ring', 'red skin', 'hair censor', 'one-eyed', 'chest hair', 'pink bowtie', 'colored tips', 'hair slicked back', 'narrow waist', 'kitsune', 'leaf on head', 'goat horns', 'fat', 'raccoon tail', 'multicolored skin', 'polka dot bow', 'manly', 'ears through headwear', 'purple skin', 'heart earrings', 'double-parted bangs', 'dark blue hair', 'big hair', 'frilled hairband', 'bonnet', 'hair over breasts', 'blank eyes', 'earphones', 'hair spread out', 'lion ears', 'genderswap \(ftm\)', 'sparkling eyes', 'tiger tail', 'cow girl', 'erune', 'huge ahoge', 'tassel earrings', 'star hat ornament', 'braided bun', 'assertive female', 'grey headwear', 'yin yang', 'diamond \(shape\)', 'animalization', 'mini top hat', 'arm ribbon', 'old man', 'braided bangs', 'bear ears', 'shark tail', 'red halo', 'red eyeshadow', 'sheep horns', 'insect wings', 'rimless eyewear', 'bow hairband', 'chest tattoo', 'sanpaku', 'biceps', 'skin-covered horns', 'yellow halo', 'traditional bowtie', 'anchor hair ornament', 'yellow hairband', 'giantess', 'no eyes', 'ear bow', 'cyborg', 'ball gag', 'hair pulled back', 'topknot', 'gigantic breasts', 'extra eyes', 'long braid', 'jester cap', 'yellow sclera', 'detached wings', 'solid oval eyes', 'monocle', 'black blindfold', 'pink choker', 'cube hair ornament', 'heart ahoge', 'cross-shaped pupils', 'doll joints', 'pregnant', 'sepia', 'cross hair ornament', 'pointy hair', 'very dark skin', 'aqua bow', 'official alternate hair length', 'front ponytail', 'pink hairband', 'bandaid on nose', 'skull hair ornament', 'heart print', 'side braids', 'tail bow', 'cross earrings', 'horn ribbon', 'cow tail', 'floppy ears', 'two-tone skin', 'plaid bow', 'purple lips', 'single mechanical arm', 'single sidelock', 'robot joints', 'solid circle eyes', 'yellow headwear', 'leg tattoo', 'faceless female', 'single wing', 'tomboy', 'belt collar', 'brown bow', 'medium bangs', 'red wings', 'monster boy', 'mismatched pupils', 'blue butterfly', 'oripathy lesion \(arknights\)', 'cowboy hat', 'flower-shaped pupils', 'brown fur', 'bird tail', 'zombie', 'gradient eyes', 'albino', 'flag print', 'unmoving pattern', 'bursting breasts', 'female pervert', 'animal ear headphones', 'pearl necklace', 'large bow', 'tail ribbon', 'snout', 'fat man', 'yandere', 'bird ears', 'pink skin', 'cat boy', 'shark girl', 'amputee', 'mouse girl', 'arthropod girl', 'fur hat', 'fur-trimmed headwear', 'interface headset', 'whiskers', 'black skin', 'frilled hat', 'striped ribbon', 'demon', 'waist bow', 'fedora', 'super crown', 'low twin braids', 'skinny', 'serval print', 'crazy eyes', 'cat hair ornament', 'reverse trap', 'ambiguous gender', 'blue wings', 'ringlets', 'butterfly wings', 'multiple hair bows', 'demon boy', 'red scrunchie', 'dragon wings', 'yellow fur', 'forked eyebrows', 'metal collar', 'armpit hair', 'scar on chest', 'hairpods', 'purple hairband', 'baby', 'body markings', 'lightning bolt symbol', 'multiple wings', 'wrist ribbon', 'red pupils', 'pirate hat', 'towel on head', 'orange bowtie', 'orange headwear', 'bow-shaped hair', 'against glass', 'oppai loli', 'heart in eye', 'leg hair', 'mini wings', 'multiple horns', 'carrot hair ornament', 'long eyelashes', 'winged arms', 'heart necklace', 'stomach tattoo', 'leaf print', 'backwards hat', 'black tail', 'red headband', 'earclip', 'earpiece', 'tiger girl', 'mechanical wings', 'white horns', 'pince-nez', 'musical note hair ornament', 'black fur', 'heart choker', 'orange ribbon', 'scar on arm', 'heart-shaped eyewear', 'circle', 'small horns', 'bat print', 'hair beads', 'uneven eyes', 'cat paws', 'white feathers', 'lion tail', 'dangle earrings', 'food print', 'print bow', 'dog boy', 'number tattoo', 'cat hood', 'raccoon girl', 'blue scrunchie', 'lion girl', 'opaque glasses', 'red sclera', 'robot ears', 'christmas ornaments', 'framed breasts', 'bandaid on cheek', 'wizard hat', 'scar on nose', 'cat ear headphones', 'tomoe \(symbol\)', 'quad tails', 'leopard print', 'rose print', 'bandage over one eye', 'sheep ears', 'animal feet', 'diagonal bangs', 'wing hair ornament', 'cowlick', 'perky breasts', 'reindeer antlers', 'bone hair ornament', 'striped tail', 'cuts', 'medical eyepatch', 'braided hair rings', 'multicolored wings', 'rectangular eyewear', 'purple wings', 'breast tattoo', 'kogal', 'squirrel ears', 'chain necklace', 'miniboy', 'ear ribbon', 'black headband', 'mole on thigh', 'multiple earrings', 'single hair intake', 'sheep girl', 'updo', 'bat hair ornament', 'goggles on headwear', 'horned headwear', 'neck tattoo', 'white scrunchie', 'talons', 'red eyeliner', 'wiffle gag', 'black scrunchie', 'white headband', 'blue-framed eyewear', 'bangs pinned back', 'squirrel tail', 'trefoil', 'horn bow', 'gas mask', 'green hairband', 'horizontal pupils', 'wolf boy', 'horseshoe ornament', 'chef hat', 'heart tattoo', 'mechanical parts', 'dragon print', 'black lips', 'earbuds', 'qingdai guanmao', 'fox boy', 'multi-tied hair', 'corruption', 'slime girl', 'animal ear piercing', 'shark hair ornament', 'colored tongue', 'bird girl', 'gold earrings', 'tassel hair ornament', 'triangle', 'feather hair', 'super saiyan', 'puckered lips', 'orange hairband', 'ankle ribbon', 'flower earrings', 'grey horns', 'crescent earrings', 'yellow pupils', 'drill sidelocks', 'pink scrunchie', 'strap between breasts', 'winged hat', 'ghost tail', 'eye mask', 'porkpie hat', 'parted hair', 'squirrel girl', 'spade \(shape\)', 'police hat', 'over-rim eyewear', 'diagonal-striped bow', 'blue fur', 'shower head', 'mohawk', 'monkey tail', 'energy wings', 'wide ponytail', 'snowflake hair ornament', 'snowflake print', 'head-mounted display', 'beard stubble', 'mechanical legs', 'yellow scrunchie', 'brown ribbon', 'jackal ears', 'whistle around neck', 'heart in mouth', 'bandaged head', 'high side ponytail', 'black feathers', 'blue lips', 'orc', 'o-ring choker', 'clover hair ornament', 'chinese knot', 'diamond-shaped pupils', 'long pointy ears', 'frilled ribbon', 'enpera', 'flame-tipped tail', 'tiger boy', 'hair horns', 'pawpads', 'skin fangs', 'deer ears', 'dirty face', 'looking over eyewear', 'pink-framed eyewear', 'feather earrings', 'broken horn', 'laurel crown', 'loose hair strand', 'large hat', 'flaming eye', 'pom pom hair ornament', 'ear tag', 'grey fur', 'grey bow', 'single ear cover', 'disembodied head', 'narrowed eyes', 'no eyewear', 'hexagram', 'yellow skin', 'crossed bandaids', 'bit gag', 'orange scrunchie', 'power symbol', 'back tattoo', 'aqua ribbon', 'large tail', 'dreadlocks', 'smiley face', 'character hair ornament', 'mechanical horns', 'humanoid robot', 'grey-framed eyewear', 'sun symbol', 'star halo', 'striped horns', 'multiple moles', 'curtained hair', 'cat hat', 'cat print', 'green lips', 'character print', 'shako cap', 'diadem', 'buzz cut', 'mole on neck', 'butterfly print', 'dragon boy', 'tam o\' shanter', 'alternate headwear', 'asymmetrical horns', 'magatama necklace', 'star choker', 'short bangs', 'orange-tinted eyewear', 'cracked skin', 'yellow-framed eyewear', 'bandage on face', 'starry sky print', 'snake tail', 'prayer beads', 'alternate form', 'fewer digits', 'afro', 'white-framed eyewear', 'tail bell', 'd-pad hair ornament', 'tri tails', 'spread wings', 'school hat', 'tall female', 'bisexual female', 'cone horns', 'buck teeth', 'cherry blossom print', 'helm', 'pink pupils', 'plaid necktie', 'hair through headwear', 'horned helmet', 'mechanical tail', 'prehensile hair', 'skull print', 'seigaiha', 'poke ball print', 'patchwork skin', 'emoji', 'octarian', 'constellation print', 'blue eyeshadow', 'star necklace', 'drop earrings', 'barcode tattoo', 'two-tone ribbon', 'bear hair ornament', 'bowl hat', 'gold hairband', 'spider girl', 'red-tinted eyewear', 'eyebrow cut', 'animal ear headwear', 'goat ears', 'single hair ring', 'cloud print', 'orange fur', 'fish hair ornament', 'dixie cup hat', 'leopard ears', 'scar on neck', 'lip piercing', 'clover print', 'ushanka', 'mars symbol', 'skull earrings', 'multicolored fur', 'party hat', 'blue horns', 'eyebrow piercing', 'bird legs', 'plaid headwear', 'white tail', 'brown hairband', 'fiery hair', 'strawberry print', 'red fur', 'green halo', 'dyed bangs', 'two-tone eyes', 'wrinkled skin', 'fake antlers', 'bat ears', 'black halo', 'bowl cut', 'bear girl', 'fangs out', 'blue headband', 'poke ball symbol', 'pink fur', 'yellow wings', 'shortstack', 'fish girl', 'fake wings', 'x-shaped pupils', 'fake facial hair', 'flower ornament', 'pillbox hat', 'circle cut', 'yellow horns', 'goblin', 'body hair', 'hair ears', 'alternate skin color', 'bow earrings', 'no wings', 'very long fingernails', 'doughnut hair bun', 'sticker on face', 'green-framed eyewear', 'pacifier', 'crescent facial mark', 'x', 'magnifying glass', 'stitched face', 'old woman', 'eyewear on headwear', 'brown horns', 'respirator', 'plant girl', 'pink eyeshadow', 'blue sclera', 'multiple braids', 'magatama earrings', 'brown-framed eyewear', 'blue-tinted eyewear', 'moon \(ornament\)', 'cow boy', 'spiked tail', 'purple eyeshadow', 'body freckles', '\. \.', 'multicolored bow', 'heart tail', 'mole on ass', 'large wings', 'triangle earrings', 'rabbit boy', 'horns through headwear', 'mechanical hands', 'purple-tinted eyewear', 'venus symbol', 'purple fur', 'unusually open eyes', 'sunflower hair ornament', 'bruise on face', 'lizard tail', 'multicolored horns', 'sakuramon', 'arm between breasts', 'two-tone headwear', 'panda ears', 'fake mustache', 'expressive hair', 'purple tail', 'drawing bow', 'object through head', 'pink wings', 'blue pupils', 'transparent wings', 'purple horns', 'phoenix crown', 'artificial eye', 'grey ribbon', 'striped headwear', 'goat girl', 'tulip hat', 'crystal hair', 'aqua headwear', 'arched bangs', 'broken halo', 'mechanical ears', 'brown wings', 'leopard tail', 'grey halo', 'no eyebrows', 'notched ear', 'monkey ears', 'pink-tinted eyewear', 'fiery horns', 'uneven horns', 'jaguar ears', 'purple halo', 'sphere earrings', 'bat girl', 'candy hair ornament', 'werewolf', 'hand tattoo', 'combat helmet', 'brushing another\'s hair', 'tapir tail', 'dark halo', 'ruffling hair', 'diving mask on head', 'triangle hair ornament', 'mechanical eye', 'spiked choker', 'sword on back', 'arrow through heart', 'scar on leg', 'huge bow', 'robot girl', 'sleeve bow', 'rabbit-shaped pupils', 'dice hair ornament', 'fish print', 'button eyes', 'chocolate on breasts', 'prehensile tail', 'multicolored headwear', 'green wings', 'looking at breasts', 'solid eyes', 'thick lips', 'compass rose halo', 'brown tail', 'strawberry hair ornament', 'food-themed earrings', 'split ponytail', 'two-tone bow', 'neck tassel', 'lion boy', 'two-tone hairband', 'wig', 'gradient skin', 'anchor print', 'polka dot headwear', 'purple scrunchie', 'glowing wings', 'crystal earrings', 'liquid hair', 'orange skin', 'cetacean tail', 'glowing hair', 'smokestack hair ornament', 'panties on head', 'crocodilian tail', 'long tail', 'legs over head', 'pearl earrings', 'glowing horns', 'red tail', 'print headwear', 'egg hair ornament', 'side drill', 'blue tail', 'huge eyebrows', 'hair wings', 'snake hair', 'thick eyelashes', 'swim cap', 'grey tail', 'choppy bangs', 'aviator sunglasses', 'pill earrings', 'no tail', 'pink tail', 'owl ears', 'pointy breasts', 'hat over one eye', 'full beard', 'bandaid hair ornament', 'footwear ribbon', 'grey hairband', 'coin hair ornament', 'bucket hat', 'alpaca ears', 'yellow tail', 'low-tied sidelocks', 'weasel ears', 'wrist bow', 'grey wings', 'pursed lips', 'no eyepatch', 'deer girl', 'white headdress', 'green tail', 'wing ornament', 'mismatched eyebrows', 'sleeve ribbon', 'purple-framed eyewear', 'rainbow hair', 'hedgehog ears', 'sideways hat', 'flower on head', 'coke-bottle glasses', 'fish boy', 'head chain', 'radiation symbol', 'orange tail', 'bandaid on forehead', 'hard hat', 'green sclera', 'hair on horn', 'ribbon-trimmed headwear', 'multiple heads', 'joestar birthmark', 'flower over eye', 'yellow-tinted eyewear', 'otter ears', 'dashed eyes', 'low-braided long hair', 'arm above head', 'lace-trimmed hairband', 'four-leaf clover hair ornament', 'potara earrings', 'detached hair', 'cephalopod eyes', 'long beard', 'camouflage headwear', 'japari bun', 'star ornament', 'striped hairband', 'hat with ears', 'bunching hair', 'ears visible through hair', 'green scrunchie', 'thick mustache', 'diamond hairband', 'polka dot scrunchie', 'cherry hair ornament', 'bear tail', 'jaguar tail', 'v-shaped eyes', 'rabbit hat', 'thick beard', 'hugging tail', 'no mole', 'green-tinted eyewear', 'ornament', 'diamond hair ornament', 'wavy eyes', 'shell hair ornament', 'heart-shaped eyes', 'chain headband', 'planet hair ornament', 'pearl hair ornament', 'multicolored hairband', 'drop-shaped pupils', 'polka dot ribbon', 'ribbon braid', 'alternate wings', 'hollow eyes', 'unworn eyepatch', 'food on breasts', 'spaceship hair ornament', 'bowler hat', 'green eyeshadow', 'pumpkin hair ornament', 'spiked hairband', 'flower in eye', 'magical boy', 'behind-the-head headphones', 'plaid ribbon', 'skull ornament', 'bear boy', 'holly hair ornament', 'uneven twintails', 'folded hair', 'pig ears', 'metal skin', 'pumpkin hat', 'cut bangs', 'mole under each eye', 'clock eyes', 'reptile girl', 'hair between breasts', 'alternate hair ornament', 'licking ear', 'braiding hair', 'hexagon hair ornament', 'tri braids', 'animal ear hairband', 'clothed male nude male', 'penis over eyes', 'solid circle pupils', 'penis to breast', 'frog girl', 'curly eyebrows', 'star-shaped eyewear', 'fiery wings', 'orange headband', 'scratching head', 'bloodshot eyes', 'green horns', 'green headband', 'single head wing', 'animal head', 'bulging eyes', 'deer tail', 'weasel girl', 'brown lips', 'lifebuoy ornament', 'frilled headwear', 'cable tail', 'safety glasses', 'leopard girl', 'wing ears', 'spade hair ornament', 'white halo', 'weasel tail', 'propeller hair ornament', 'wide oval eyes', 'otter tail', 'pom pom earrings', 'checkered bow', 'fruit hat ornament', 'starfish hair ornament', 'aqua hairband', 'crystal wings', 'object head', 'multicolored tail', 'gradient wings', 'giant male', 'purple pupils', 'torn wings', 'head on head', 'moose ears', 'pointy hat', 'hair over one breast', 'arm over head', 'grabbing another\'s ear', 'forked tail', 'lightning bolt hair ornament', 'undone neck ribbon', 'hedgehog tail', 'lop rabbit ears', 'sparse chest hair', 'pink horns', 'pokemon ears', 'ankle bow', 'bird boy', 'bandaid on head', 'implied extra ears', 'hat tassel', 'fruit on head', 'starry hair', 'sparkle hair ornament', 'long ribbon', 'rice hat', 'washing hair', 'anchor earrings', 'asymmetrical sidelocks', 'mini witch hat', 'unworn hair ornament', 'heart hair', 'arthropod boy', 'detached ahoge', 'large ears', 'aviator cap', 'monkey boy', 'female service cap', 'moth girl', 'glove bow', 'bangs', 'shiny hair', 'light purple hair', 'oni horns', 'pillow hat', 'polos crown', 'light green hair', 'monocle hair ornament', 'dark green hair', 'pouty lips', 'bunny-shaped pupils', 'bunny hat'
}

# 內建的衣服標籤名單
clothing_tags = {
    'shirt', 'skirt', 'long sleeves', 'hair ornament', 'gloves', 'dress', 'thighhighs', 'hat', 'bow', 'navel', 'ribbon', 'cleavage', 'jewelry', 'bare shoulders', 'underwear', 'jacket', 'school uniform', 'collarbone', 'white shirt', 'panties', 'swimsuit', 'hair ribbon', 'short sleeves', 'hair bow', 'pantyhose', 'earrings', 'bikini', 'pleated skirt', 'frills', 'hairband', 'boots', 'open clothes', 'necktie', 'detached sleeves', 'shorts', 'japanese clothes', 'shoes', 'sleeveless', 'black gloves', 'alternate costume', 'collared shirt', 'choker', 'barefoot', 'socks', 'glasses', 'pants', 'serafuku', 'puffy sleeves', 'hairclip', 'belt', 'black thighhighs', 'elbow gloves', 'midriff', 'white gloves', 'bowtie', 'hood', 'black skirt', 'hair flower', 'official alternate costume', 'wide sleeves', 'miniskirt', 'fingerless gloves', 'black footwear', 'kimono', 'white dress', 'holding weapon', 'off shoulder', 'necklace', 'striped clothes', 'nail polish', 'star \(symbol\)', 'bag', 'black dress', 'scarf', 'cape', 'white thighhighs', 'bra', 'armor', 'vest', 'open jacket', 'halo', 'apron', 'red bow', 'white panties', 'leotard', 'coat', 'black jacket', 'high heels', 'collar', 'sweater', 'bracelet', 'uniform', 'red ribbon', 'crop top', 'black shirt', 'puffy short sleeves', 'blue skirt', 'black pantyhose', 'neckerchief', 'sleeves past wrists', 'fur trim', 'see-through', 'wrist cuffs', 'maid', 'strapless', 'zettai ryouiki', 'clothing cutout', 'black headwear', 'plaid', 'torn clothes', 'one-piece swimsuit', 'sash', 'maid headdress', 'sleeveless shirt', 'short shorts', 'bare arms', 'sleeveless dress', 'ascot', 'black panties', 'cosplay', 'kneehighs', 'bare legs', 'thigh strap', 'black bow', 'covered navel', 'hoodie', 'neck ribbon', 'black ribbon', 'detached collar', 'tattoo', 'black choker', 'dress shirt', 'buttons', 'open shirt', 'sideboob', 'bell', 'military', 'mask', 'skindentation', 'capelet', 'bodysuit', 'blue dress', 'black pants', 'no bra', 'black bikini', 'white headwear', 'red skirt', 'blue bow', 'turtleneck', 'underboob', 'witch hat', 'highleg', 'military uniform', 'headband', 'black shorts', 'beret', 'side-tie bikini bottom', 'brown footwear', 'halterneck', 'chain', 'playboy bunny', 'headphones', 'piercing', 'white jacket', 'holding sword', 'white socks', 'blush stickers', 'chinese clothes', 'white bikini', 'no shoes', 'plaid skirt', 'thigh boots', 'white footwear', 'headgear', 'sandals', 'floral print', 'garter straps', 'short dress', 'sunglasses', 'obi', 'red dress', 'hood down', 'frilled dress', 'cleavage cutout', 'white skirt', 'blue shirt', 'hair tubes', 'ring', 'holding food', 'bound', 'blue jacket', 'black socks', 'black hairband', 'eyepatch', 'scrunchie', 'white bow', 'formal', 'mob cap', 'cardigan', 'backpack', 'frilled skirt', 'tank top', 'blazer', 'suspenders', 'helmet', 'suit', 'feathers', 'x hair ornament', 'underwear only', 'blue ribbon', 'frilled sleeves', 'school swimsuit', 'cross', 'hat ribbon', 'denim', 'crown', 'knee boots', 'pink bow', 'red necktie', 'tiara', 'juliet sleeves', 'polka dot', 'black nails', 'ear piercing', 'wing collar', 'lingerie', 'animal print', 'red shirt', 'undressing', 'striped thighhighs', 'blue sailor collar', 'sneakers', 'black leotard', 'white border', 't-shirt', 'tassel', 'holding gun', 'gem', 'red footwear', 'white apron', 'bondage', 'red bowtie', 'hair bobbles', 'lipstick', 'green skirt', 'goggles', 'shoulder armor', 'holding cup', 'brooch', 'black bra', 'fishnets', 'loafers', 'crescent', 'towel', 'single thighhigh', 'pink dress', 'strapless leotard', 'hat bow', 'grey shirt', 'black necktie', 'no pants', 'eyewear on head', 'bike shorts', 'hooded jacket', 'armband', 'casual', 'revealing clothes', 'red headwear', 'gauntlets', 'white ribbon', 'rope', 'sheath', 'china dress', 'ribbon trim', 'pink panties', 'adapted costume', 'multicolored clothes', 'wristband', 'hakama', 'blouse', 'puffy long sleeves', 'veil', 'red jacket', 'red nails', 'lace trim', 'waist apron', 'skirt set', 'pelvic curtain', 'strapless dress', 'baseball cap', 'string bikini', 'striped panties', 'blue headwear', 'bridal gauntlets', 'cloak', 'peaked cap', 'highleg leotard', 'red neckerchief', 'purple dress', 'side-tie panties', 'semi-rimless eyewear', 'white pantyhose', 'jingle bell', 'hand fan', 'grey skirt', 'front-tie top', 'bow panties', 'buckle', 'clothes writing', 'pom pom \(clothes\)', 'micro bikini', 'yellow bow', 'maid apron', 'sleeves past fingers', 'hood up', 'corset', 'neck bell', 'blue nails', 'skin tight', 'o-ring', 'hakama skirt', 'black belt', 'lace', 'holding phone', 'no headwear', 'tokin hat', 'white sleeves', 'cropped jacket', 'bikini top only', 'brown gloves', 'restrained', 'red gloves', 'mary janes', 'spikes', 'blue bikini', 'holding book', 'side slit', 'black coat', 'camisole', 'strap slip', 'armlet', 'green bow', 'hair scrunchie', 'sleeves rolled up', 'gold trim', 'blue necktie', 'santa hat', 'black sailor collar', 'pink skirt', 'single glove', 'pink ribbon', 'white sailor collar', 'zipper', 'open coat', 'blue shorts', 'pencil skirt', 'pink shirt', 'pendant', 'ribbed sweater', 'topless male', 'high heel boots', 'track jacket', 'single earring', 'frilled apron', 'asymmetrical legwear', 'sweater vest', 'cross-laced footwear', 'headset', 'black vest', 'frilled bikini', 'beads', 'pocket', 'vertical-striped clothes', 'green dress', 'unworn headwear', 'frilled shirt collar', 'black-framed eyewear', 'brown pantyhose', 'thong', 'red bikini', 'purple bow', 'long skirt', 'high-waist skirt', 'round eyewear', 'blue footwear', 'crossdressing', 'white bra', 'cuffs', 'gym uniform', 'purple skirt', 'yellow shirt', 'goggles on head', 'black bowtie', 'red-framed eyewear', 'epaulettes', 'ribbon-trimmed sleeves', 'santa costume', 'high collar', 'brown jacket', 'denim shorts', 'brown skirt', 'hat ornament', 'panties under pantyhose', 'buruma', 'white pants', 'school bag', 'nontraditional miko', 'pouch', 'thighband pantyhose', 'black serafuku', 'dual wielding', 'red scarf', 'fur collar', 'green jacket', 'sailor dress', 'robe', 'garter belt', 'white shorts', 'pauldrons', 'competition swimsuit', 'yellow ribbon', 'lolita fashion', 'top hat', 'sun hat', 'white hairband', 'watch', 'blood on face', 'blue one-piece swimsuit', 'turtleneck sweater', 'star hair ornament', 'white kimono', 'sports bra', 'under-rim eyewear', 'grey jacket', 'shiny clothes', 'white coat', 'striped shirt', 'impossible clothes', 'jeans', 'circlet', 'belt buckle', 'holding umbrella', 'shoulder bag', 'green shirt', 'partially fingerless gloves', 'sarashi', 'striped bikini', 'crystal', 'shawl', 'bandaged arm', 'hairpin', 'hoop earrings', 'sheathed', 'unworn hat', 'blue bowtie', 'green headwear', 'yukata', 'mini hat', 'white leotard', 'purple shirt', 'military hat', 'breastplate', 'pajamas', 'brown headwear', 'black sleeves', 'blue pants', 'bespectacled', 'holding staff', 'shirt tucked in', 'striped bow', 'tress ribbon', 'mouth mask', 'handbag', 'blue panties', 'animal hood', 'hugging object', 'collared dress', 'scar across eye', 'backless outfit', 'meme attire', 'tabard', 'long dress', 'sportswear', 'torn pantyhose', 'fishnet pantyhose', 'ofuda', 'off-shoulder dress', 'forehead mark', 'front-tie bikini top', 'pink footwear', 'heart hair ornament', 'lab coat', 'panties aside', 'black one-piece swimsuit', 'red hairband', 'hair bell', 'skirt hold', 'wedding dress', 'blue kimono', 'bandaid on face', 'nurse cap', 'purple ribbon', 'holding clothes', 'bloomers', 'butterfly hair ornament', 'red cape', 'hat flower', 'hair tie', 'blue gloves', 'arrow \(symbol\)', 'purple nails', 'miko', 'holding flower', 'pasties', 'straw hat', 'green ribbon', 'bandana', 'black bodysuit', 'blue leotard', 'visor cap', 'winter uniform', 'covering own mouth', 'orange bow', 'drawstring', 'yellow ascot', 'red vest', 'stud earrings', 'blindfold', 'fur-trimmed jacket', 'brown belt', 'off-shoulder shirt', 'pink jacket', 'center frills', 'shrug \(clothing\)', 'panties around one leg', 'black cape', 'pink bra', 'criss-cross halter', 'anklet', 'center opening', 'white sweater', 'headpiece', 'gag', 'suspender skirt', 'highleg panties', 'beanie', 'grabbing from behind', 'spaghetti strap', 'bandeau', 'white scarf', 'pink bikini', 'folding fan', 'underbust', 'holding knife', 'back bow', 'white one-piece swimsuit', 'emblem', 'adjusting eyewear', 'double-breasted', 'brown thighhighs', 'eighth note', 'gakuran', 'geta', 'sailor hat', 'nun', 'black collar', 'tabi', 'name tag', 'cabbie hat', 'microskirt', 'pinafore dress', 'anchor symbol', 'arm warmers', 'petticoat', 'frog hair ornament', 'arm tattoo', 'sailor shirt', 'bodystocking', 'highleg swimsuit', 'frilled shirt', 'tracen school uniform', 'habit', 'body fur', 'yellow jacket', 'unbuttoned', 'winter clothes', 'bikini under clothes', 'wristwatch', 'lowleg', 'bound wrists', 'summer uniform', 'gothic lolita', 'nipple slip', 'cow print', 'brown shirt', 'lace-up boots', 'cheerleader', 'checkered clothes', 'holding bag', 'grey pants', 'food-themed hair ornament', 'brown dress', 'swimsuit under clothes', 'taut clothes', 'hitodama', 'witch', 'vambraces', 'overalls', 'holding bottle', 'pom pom \(cheerleading\)', 'pink headwear', 'wrist scrunchie', 'black sweater', 'mittens', 'blue thighhighs', 'holding poke ball', 'military vehicle', 'hair stick', 'ankle boots', 'layered sleeves', 'toeless legwear', 'print bikini', 'food in mouth', 'headdress', 'breast pocket', 'print kimono', 'feather hair ornament', 'gohei', 'tinted eyewear', 'blood on clothes', 'wedding ring', 'bangle', 'cable', 'zipper pull tab', 'purple bikini', 'unzipped', 'sundress', 'pleated dress', 'purple jacket', 'crotch seam', 'armored dress', 'armored boots', 'mask on head', 'midriff peek', 'red kimono', 'black kimono', 'ninja', 'purple gloves', 'grey dress', 'bridal garter', 'tube top', 'bare back', 'holding tray', 'horn ornament', 'holding fan', 'halloween costume', 'open fly', 'mini crown', 'nurse', 'white capelet', 'animal costume', 'shoulder cutout', 'naked shirt', 'half gloves', 'blue hairband', 'brown pants', 'red choker', 'strap gap', 'japanese armor', 'male underwear', 'black hoodie', 'uneven legwear', 'blue halo', 'falling petals', 'blue vest', 'haori', 'office lady', 'holster', 'waitress', 'thighlet', 'holding polearm', 'earmuffs', 'red thighhighs', 'purple thighhighs', 'open cardigan', 'yellow bikini', 'two-tone dress', 'bow bra', 'naval uniform', 'bridal veil', 'purple headwear', 'flat cap', 'pilot suit', 'paw gloves', 'tight clothes', 'randoseru', 'holding microphone', 'red panties', 'blue neckerchief', 'rigging', 'ear covers', 'triangular headpiece', 'snake hair ornament', 'star print', 'paw print', 'pink kimono', 'purple panties', 'fishnet thighhighs', 'heart of string', 'shibari', 'red hakama', 'muneate', 'yellow dress', 'sarong', 'red shorts', 'slippers', 'mismatched legwear', 'mechanical halo', 'highleg bikini', 'bracer', 'yellow neckerchief', 'covering crotch', 'cropped shirt', 'hakama short skirt', 'sleeve cuffs', 'red ascot', 'hip vent', 'animal hat', 'sack', 'arm strap', 'navel cutout', 'kita high school uniform', 'bare pectorals', 'scar on cheek', 'white cape', 'pink halo', 'white choker', 'partially unbuttoned', 'unworn eyewear', 'shoulder tattoo', 'condom wrapper', 'lolita hairband', 'facepaint', 'knee pads', 'backless dress', 'green nails', 'casual one-piece swimsuit', 'torn thighhighs', 'print shirt', 'animal nose', 'tied shirt', 'layered dress', 'string panties', 'bandaged leg', 'sleeveless turtleneck', 'bikini skirt', 'red leotard', 'layered skirt', 'leggings', 'green bikini', 'star earrings', 'crescent hair ornament', 'scabbard', 'brown coat', 'holding animal', 'pantyhose under shorts', 'holding instrument', 'tasuki', 'ribbon choker', 'lace-trimmed panties', 'lace-trimmed legwear', 'leg ribbon', 'tate eboshi', 'plugsuit', 'hat feather', 'garrison cap', 'short kimono', 'magatama', 'parasol', 'fur-trimmed coat', 'leather', 'asymmetrical clothes', 'lace-trimmed bra', 'head wreath', 'shimenawa', 'frilled bow', 'tilted headwear', 'single bare shoulder', 'animal on head', 'yellow skirt', 'sample watermark', 'holding hair', 'purple footwear', 'blue bra', 'raglan sleeves', 'harness', 'clothes around waist', 'tiger print', 'blue scarf', 'grey footwear', 'asymmetrical gloves', 'ear ornament', 'o-ring top', 'asymmetrical wings', 'naked towel', 'o-ring bikini', 'forehead jewel', 'stitches', 'white collar', 'facial tattoo', 'purple kimono', 'headphones around neck', 'pantyhose pull', 'white tank top', 'cross necklace', 'french kiss', 'crescent hat ornament', 'crop top overhang', 'short over long sleeves', 'black scarf', 'open kimono', 'oversized clothes', 'black neckerchief', 'two-sided fabric', 'holding bouquet', 'strap', 'fur-trimmed sleeves', 'platform footwear', 'sakuragaoka high school uniform', 'one-piece tan', 'white hoodie', 'striped dress', 'orange shirt', 'blue cape', 'holding gift', 'pectoral cleavage', 'white belt', 'leaf hair ornament', 'holding stuffed toy', 'handcuffs', 'red pants', 'navel piercing', 'rabbit hair ornament', 'spiked collar', 'shackles', 'neck ring', 'pink bowtie', 'brown sweater', 'pants pull', 'yellow necktie', 'holding fruit', 'fox mask', 'belt pouch', 'spiked bracelet', 'unworn shoes', 'red coat', 'black tank top', 'grey shorts', 'head scarf', 'greaves', 'leaf on head', 'necktie between breasts', 'micro shorts', 'slingshot swimsuit', 'camouflage', 'east asian architecture', 'arm cannon', 'polka dot bow', 'striped necktie', 'naked apron', 'red capelet', 'green vest', 'bandaid on leg', 'white bowtie', 'fur-trimmed gloves', 'no shirt', 'loose socks', 'body writing', 'badge', 'multicolored jacket', 'shorts under skirt', 'heart earrings', 'black capelet', 'bead necklace', 'babydoll', 'arm guards', 'green necktie', 'ooarai school uniform', 'brown shorts', 'undershirt', 'frilled hairband', 'grey sweater', 'orange skirt', 'see-through sleeves', 'hooded cloak', 'bonnet', 'green shorts', 'huge weapon', 'blue choker', 'skirt pull', 'brown cardigan', 'gym shirt', 'earphones', 'bra lift', 'blue coat', 'bead bracelet', 'fundoshi', 'iron cross', 'cutoffs', 'white ascot', 'smoking pipe', 'cow girl', 'bikini pull', 'hachimaki', 'white bodysuit', 'plaid vest', 'pink gloves', 'shoulder pads', 'argyle clothes', 'tassel earrings', 'drink can', 'bikini armor', 'black sports bra', 'yellow bowtie', 'pink thighhighs', 'star hat ornament', 'latex', 'holding chopsticks', 'purple bowtie', 'torn shirt', 'animal collar', 'grey headwear', 'sweater dress', 'yin yang', 'bobby socks', 'grey gloves', 'blue sweater', 'diamond \(shape\)', 'fringe trim', 'green footwear', 'striped pantyhose', 'mini top hat', 'arm garter', 'aqua necktie', 'frilled panties', 'off-shoulder sweater', 'arm ribbon', 'fur-trimmed capelet', 'broom riding', 'untied bikini', 'police', 'holding spoon', 'holding bow \(weapon\)', 'red collar', 'ear blush', 'object on head', 'multicolored dress', 'single shoe', 'bikini bottom only', 'dirty', 'red halo', 'yellow footwear', 'green kimono', 'vertical-striped thighhighs', 'red sweater', 'heart brooch', 'rimless eyewear', 'bow hairband', 'male swimwear', 'frilled bra', 'blue gemstone', 'chest tattoo', 'holding fork', 'green panties', 'grey thighhighs', 'zouri', 'collared jacket', 'race queen', 'fur-trimmed dress', 'yellow halo', 'traditional bowtie', 'anchor hair ornament', 'yellow hairband', 'ear bow', 'unworn panties', 'fur-trimmed cape', 'long coat', 'ball gag', 'v-neck', 'chest jewel', 'unworn skirt', 'extra arms', 'topknot', 'blue hoodie', 'sailor senshi uniform', 'cum string', 'between fingers', 'sleeveless jacket', 'green pants', 'white neckerchief', 'superhero', 'single sock', 'brown vest', 'print panties', 'waist cape', 'shirt pull', 'green gloves', 'holding card', 'holding can', 'hooded coat', 'clothed male nude female', 'jester cap', 'open vest', 'plaid shirt', 'weapon on back', 'holding cigarette', 'monocle', 'black blindfold', 'pink sweater', 'track suit', 'pink choker', 'pillow hug', 'cube hair ornament', 'frilled collar', 'floating object', 'black suit', 'nightgown', 'frilled choker', 'striped bowtie', 'white necktie', 'jumpsuit', 'faulds', 'police uniform', 'uwabaki', 'bride', 'blood on hands', 'downblouse', 'yellow nails', 'holding wand', 'kariginu', 'sepia', 'cross hair ornament', 'weapon over shoulder', 'competition school swimsuit', 'over-kneehighs', 'aqua bow', 'leotard under clothes', 'lowleg panties', 'pink hairband', 'sweater lift', 'taut shirt', 'bandaid on nose', 'skull hair ornament', 'red bodysuit', 'heart print', 'heart cutout', 'plaid scarf', 'side braids', 'holding hat', 'open hoodie', 'charm \(object\)', 'red bra', 'purple bra', 'chest harness', 'adjusting headwear', 'short necktie', 'blood splatter', 'cross earrings', 'holding candy', 'coat on shoulders', 'halter dress', 'tokiwadai school uniform', 'horn ribbon', 'multiple rings', 'dog tags', 'two-tone skin', 'plaid bow', 'dougi', 'chaldea uniform', 'leg warmers', 'loincloth', 'frilled thighhighs', 'bra pull', 'pinstripe pattern', 'purple necktie', 'military jacket', 'meat', 'forehead protector', 'holding paper', 'two-tone shirt', 'holding ball', 'yellow headwear', 'lanyard', 'otonokizaka school uniform', 'leg tattoo', 'sleeves past elbows', 'heart pasties', 'bound legs', 'white camisole', 'thighhighs under boots', 'belt collar', 'brown bow', 'thigh holster', 'bubble skirt', 'striped socks', 'blue butterfly', 'waistcoat', 'cowboy hat', 'print dress', 'asymmetrical sleeves', 'polka dot panties', 'lapels', 'blue bodysuit', 'knight', 'torn pants', 'star in eye', 'holding removed eyewear', 'print pantyhose', 'holding pen', 'grey pantyhose', 'gradient eyes', 'cow print bikini', 'thong bikini', 'bra visible through clothes', 'flag print', 'unmoving pattern', 'white bloomers', 'full armor', 'panty peek', 'popped collar', 'chest sarashi', 'wet panties', 'yellow gloves', 'pearl necklace', 'large bow', 'multicolored nails', 'old school swimsuit', 'no socks', 'green bowtie', 'flower knot', 'blue sleeves', 'bird ears', 'stirrup legwear', 'maebari', 'orange jacket', 'purple leotard', 'partially unzipped', 'aiguillette', 'dolphin shorts', 'vertical-striped shirt', 'fur hat', 'obijime', 'fur-trimmed headwear', 'bra strap', 'strap pull', 'orange dress', 'blue buruma', 'interface headset', 'pink necktie', 'toeless footwear', 'crotchless', 'suit jacket', 'striped scarf', 'mismatched gloves', 'grey vest', 'blue socks', 'medium skirt', 'argyle legwear', 'frilled hat', 'tengu-geta', 'striped skirt', 'showgirl skirt', 'striped ribbon', 'waist bow', 'quiver', 'red buruma', 'falling leaves', 'closed umbrella', 'button gap', 'black cloak', 'fedora', 'cable knit', 'holding camera', 'grabbing own ass', 'backboob', 'single leg pantyhose', 'grey cardigan', 'elbow pads', 'print thighhighs', 'winter coat', 'yellow sweater', 'serval print', 'orange flower', 'see-through cleavage', 'back cutout', 'two-tone jacket', 'single elbow glove', 'cat hair ornament', 'yellow shorts', 'cake slice', 'orange bikini', 'torn dress', 'black ascot', 'print skirt', 'multiple hair bows', 'holding broom', 'track pants', 'bikini bottom aside', 'holding shield', 'towel around neck', 'blood on weapon', 'dress bow', 'pink neckerchief', 'red scrunchie', 'grey hoodie', 'business suit', 'baggy pants', 'holding bowl', 'metal collar', 'unworn mask', 'bandaged hand', 'white vest', 'hose', 'frilled hair tubes', 'impossible shirt', 'unbuttoned shirt', 'scar on chest', 'holding lollipop', 'blue capelet', 'hairpods', 'footwear bow', 'pocket watch', 'purple hairband', 'body markings', 'lightning bolt symbol', 'wrist ribbon', 'single sleeve', 'surgical mask', 'unconventional maid', 'open collar', 'bustier', 'brown bag', 'pirate hat', 'red rope', 'anal tail', 'sleeveless kimono', 'leather jacket', 'trench coat', 'orange bowtie', 'pink cardigan', 'planted sword', 'orange headwear', 'loose necktie', 'bodypaint', 'brown scarf', 'green sailor collar', 'holding towel', 'pink scarf', 'red sailor collar', 'fur-trimmed boots', 'multicolored skirt', 'gym shorts', 'carrot hair ornament', 'power armor', 'covered collarbone', 'silk', 'sailor', 'heart necklace', 'frilled gloves', 'yellow panties', 'blue ascot', 'stomach tattoo', 'underboob cutout', 'leaf print', 'arm belt', 'backwards hat', 'cat cutout', 'polka dot bikini', 'crescent pin', 'red headband', 'strapless bikini', 'red sleeves', 'earclip', 'earpiece', 'checkered skirt', 'holding box', 'white robe', 'reverse outfit', 'thorns', 'grey coat', 'cat lingerie', 'purple pantyhose', 'pince-nez', 'musical note hair ornament', 'holding mask', 'frilled pillow', 'torn skirt', 'thong leotard', 'two-tone gloves', 'red belt', 'heart choker', 'orange ribbon', 'black bag', 'scar on arm', 'heart-shaped eyewear', 'policewoman', 'ribbon-trimmed legwear', 'circle', 'hooded capelet', 'blue belt', 'bat print', 'hair beads', 'demon slayer uniform', 'soccer uniform', 'striped jacket', 'pink shorts', 'holding scythe', 'holding pom poms', 'grey sailor collar', 'purple vest', 'virgin killer sweater', 'improvised gag', 'reverse bunnysuit', 'white feathers', 'dangle earrings', 'food print', 'bike shorts under skirt', 'print bow', 'yellow scarf', 'tight pants', 'number tattoo', 'kiseru', 'sleeves pushed up', 'cat hood', 'print gloves', 'soul gem', 'glowing weapon', 'blue scrunchie', 'leotard aside', 'bikini tan', 'opaque glasses', 'holding axe', 'kunai', 'pubic hair peek', 'no legwear', 'bandaids on nipples', 'idol clothes', 'orange necktie', 'christmas ornaments', 'pink leotard', 'multiple belts', 'sideless outfit', 'flip-flops', 'bandaid on cheek', 'wizard hat', 'holding pokemon', 'aran sweater', 'goth fashion', 'single detached sleeve', 'scar on nose', 'cat ear headphones', 'employee uniform', 'plaid dress', 'tomoe \(symbol\)', 'black armor', 'jacket partially removed', 'asymmetrical footwear', 'red sash', 'vertical-striped dress', 'orange nails', 'leopard print', 'rose print', 'costume', 'green gemstone', 'bandage over one eye', 'breast curtains', 'unworn bra', 'purple bodysuit', 'ribbed shirt', 'blue pantyhose', 'spacesuit', 'pumps', 'two-tone skirt', 'swimsuit aside', 'wing hair ornament', 'green cape', 'spread toes', 'maid bikini', 'holding dagger', 'bone hair ornament', 'tail through clothes', 'green leotard', 'green scarf', 'furisode', 'presenting armpit', 'jacket around waist', 'green sweater', 'multiple crossover', 'gloved handjob', 'medical eyepatch', 'purple choker', 'swim trunks', 'holding underwear', 'obiage', 'frilled one-piece swimsuit', 'kote', 'coattails', 'braided hair rings', 'non-humanoid robot', 'orange footwear', 'yellow kimono', 'multicolored wings', 'pantylines', 'blue hakama', 'rectangular eyewear', 'slave', 'skirt suit', 'aqua bowtie', 'feather boa', 'o-ring bottom', 'purple wings', 'american flag legwear', 'clothes grab', 'letterman jacket', 'breast tattoo', 'chain necklace', 'single kneehigh', 'ear ribbon', 'barbell piercing', 'black headband', 'green coat', 'mole on thigh', 'multiple earrings', 'domino mask', 'layered clothes', 'purple pants', 'grey panties', 'kanzashi', 'wet swimsuit', 'egyptian', 'white wrist cuffs', 'frilled kimono', 'eyepatch bikini', 'bag charm', 'fur-trimmed hood', 'bat hair ornament', 'diagonal-striped clothes', 'goggles on headwear', 'burn scar', 'neck tattoo', 'paradis military uniform', 'open dress', 'white scrunchie', 'capri pants', 'bra peek', 'button badge', 'rudder footwear', 'red eyeliner', 'holding drink', 'wiffle gag', 'black scrunchie', 'white headband', 'blue-framed eyewear', 'nightcap', 'skates', 'bird on head', 'american flag dress', 'layered bikini', 'diamond button', 'print bowtie', 'black camisole', 'trefoil', 'black cardigan', 'horn bow', 'naked sweater', 'pink apron', 'okobo', 'gas mask', 'green hairband', 'two-tone swimsuit', 'rei no himo', 'yoga pants', 'yellow cardigan', 'black robe', 'horseshoe ornament', 'hand over own mouth', 'basketball \(object\)', 'green bra', 'hagoromo', 'holding smoking pipe', 'carrying person', 'tie clip', 'chef hat', 'santa dress', 'high-waist shorts', 'green thighhighs', 'uranohoshi school uniform', 'orange gloves', 'black mask', 'heart tattoo', 'rabbit hood', 'four-leaf clover', 'excalibur \(fate/stay night\)', 'tunic', 'rabbit print', 'purple sleeves', 'dragon print', 'spider web print', 'sleeveless sweater', 'ankle socks', 'earbuds', 'grey socks', 'sleepwear', 'qingdai guanmao', 'medal', 'slime girl', 'shark hair ornament', 'shibari over clothes', 'holding flag', 'ribbed dress', 'gold earrings', 'two-tone bikini', 'lace panties', 'tassel hair ornament', 'chained', 'loose belt', 'triangle', 'food on head', 'mandarin collar', 'soldier', 'sailor bikini', 'striped bra', 'orange hairband', 'ankle ribbon', 'flower earrings', 'strappy heels', 'torn sleeves', 'plunging neckline', 'flats', 'red pantyhose', 'blue serafuku', 'white cloak', 'torn bodysuit', 'crescent earrings', 'purple umbrella', 'harem outfit', 'pink scrunchie', 'hands in opposite sleeves', 'strap between breasts', 'winged hat', 'crotch rope', 'holding arrow', 'shirt tug', 'meiji schoolgirl uniform', 'purple cape', 'wa maid', 'undersized clothes', 'orange bodysuit', 'paw shoes', 'cross scar', 'suspender shorts', 'see-through dress', 'eye mask', 'female pov', 'holding controller', 'gold chain', 'microdress', 'blue cardigan', 'porkpie hat', 'breast slip', 'clitoral hood', 'white sports bra', 'thumb ring', 'spade \(shape\)', 'police hat', 'holding doll', 'white cardigan', 'over-rim eyewear', 'diagonal-striped bow', 'covering own eyes', 'heart-shaped pillow', 'hanfu', 'hugging doll', 'unworn helmet', 'multi-strapped bikini bottom', 'multicolored swimsuit', 'snowflake hair ornament', 'purple shorts', 'covered abs', 'blue overalls', 'snowflake print', 'head-mounted display', 'rubber boots', 'checkered scarf', 'holding sheath', 'g-string', 'breastless clothes', 'black apron', 'purple scarf', 'purple coat', 'american flag bikini', 'striped pants', 'jirai kei', 'black corset', 'yellow scrunchie', 'brown ribbon', 'whistle around neck', 'heart in mouth', 'side cutout', 'bandaged head', 'holding basket', 'mismatched bikini', 'black feathers', 'red socks', 'neck ruff', 'feather trim', 'pirate', 'white nails', 'tongue piercing', 'shoulder spikes', 'holding panties', 'o-ring choker', 'clover hair ornament', 'evening gown', 'scar on forehead', 'duffel bag', 'pink bodysuit', 'holding leash', 'alternate legwear', 'chinese knot', 'frilled ribbon', 'orange shorts', 'talisman', 'holding hammer', 'enpera', 'two-footed footjob', 'torn cape', 'holding condom', 'bow bikini', 'fashion', 'orange kimono', 'red armband', 'strapless shirt', 'black hakama', 'holding sign', 'shimakaze \(kancolle\) \(cosplay\)', 'pink hoodie', 'gem uniform \(houseki no kuni\)', 'asticassia school uniform', 'puffy detached sleeves', 'aqua dress', 'egyptian clothes', 'uneven gloves', 'medium dress', 'fur coat', 'looking over eyewear', 'bodysuit under clothes', 'open shorts', 'flower wreath', 'skirt tug', 'tailcoat', 'pink-framed eyewear', 'blue apron', 'chest belt', 'year of the tiger', 'red cloak', 'year of the rabbit', 'lace-trimmed dress', 'feather earrings', 'holding sack', 'raincoat', 'santa bikini', 'brown sweater vest', 'laurel crown', 'large hat', 'yugake', 'holding cat', 'pith helmet', 'brown capelet', 'open-chest sweater', 'pom pom hair ornament', 'arabian clothes', 'seamed legwear', 'ear tag', 'grey bow', 'chemise', 'aqua skirt', 'pant suit', 'hooded cape', 'single ear cover', 'wrestling outfit', 'holding paintbrush', 'cat ear panties', 'red umbrella', 'fur scarf', 'pink sleeves', 'safety pin', 'flower \(symbol\)', 'zero suit', 'bomber jacket', 'no eyewear', 'patterned clothing', 'naoetsu high school uniform', 'brown kimono', 'rice bowl', 'hexagram', 'purple sweater', 'high-waist pants', 'tuxedo', 'overskirt', 'chain leash', 'crossed bandaids', 'bit gag', 'orange scrunchie', 'negligee', 'green choker', 'cyberpunk', 'borrowed clothes', 'power symbol', 'diagonal-striped necktie', 'back tattoo', 'brown sailor collar', 'black leggings', 'hand on hilt', 'aqua ribbon', 'ainu clothes', 'yellow vest', 'smiley face', 'red hoodie', 'white suit', 'pink pants', 'bandaid on arm', 'wreath', 'carrot necklace', 'character hair ornament', 'grey-framed eyewear', 'ribbon-trimmed skirt', 'sun symbol', 'see-through legwear', 'bare hips', 'uneven sleeves', 'steam censor', 'belly chain', 'star halo', 'white male underwear', 'samurai', 'turtleneck dress', 'ankle cuffs', 'untied panties', 'aqua bikini', 'holding handheld game console', 'striped sleeves', 'cat hat', 'aqua shirt', 'cat print', 'red bag', 'kitauji high school uniform', 'st\. gloriana\'s school uniform', 'black sash', 'frilled capelet', 'character print', 'shako cap', 'diadem', 'impossible bodysuit', 'nose piercing', 'orange choker', 'boxers', 'holding clipboard', 'brown cape', 'holding syringe', 'torn shorts', 'very long sleeves', 'turban', 'transparent umbrella', 'skirt around one leg', 'butterfly print', 'mask pull', 'naked jacket', 'red one-piece swimsuit', 'oni mask', 'denim skirt', 'spandex', 'ribbed legwear', 'notched lapels', 'fanny pack', 'tam o\' shanter', 'green hoodie', 'white sash', 'magatama necklace', 'star choker', 'single pauldron', 'purple sailor collar', 'diving mask', 'crotchless panties', 'strapless bra', 'vertical-striped skirt', 'orange-tinted eyewear', 'insignia', 'yellow-framed eyewear', 'legwear garter', 'feather-trimmed sleeves', 'traditional nun', 'ooarai military uniform', 'naked coat', 'plaid bikini', 'studded belt', 'bandage on face', 'starry sky print', 'shared scarf', 'pendant choker', 'impossible leotard', 'pink bag', 'korean clothes', 'prayer beads', 'nontraditional playboy bunny', 'dirty clothes', 'holding pencil', 'shuuchiin academy school uniform', 'thigh ribbon', 'fur-trimmed legwear', 'oversized shirt', 'holding lantern', 'two-sided cape', 'assault visor', 'open skirt', 'tennis uniform', 'shark hood', 'boxing gloves', 'plaid bowtie', 'glowing sword', 'holding stick', 'panty straps', 'white-framed eyewear', 'briefs', 'multicolored bodysuit', 'team rocket', 'turnaround', 'black wristband', 'breast bondage', 'd-pad hair ornament', 'champion\'s tunic \(zelda\)', 'sweatband', 'latex bodysuit', 'yellow choker', 'skull mask', 'grey kimono', 'vertical-striped pantyhose', 'little busters! school uniform', 'blood stain', 'school hat', 'gown', 'sweater around waist', 'red armor', 'sleeping on person', 'lace bra', 'tactical clothes', 'impossible dress', 'hikarizaka private high school uniform', 'hands on headwear', 'u u', 'spiked armlet', 'holding whip', 'millennium cheerleader outfit \(blue archive\)', 'see-through leotard', 'green neckerchief', 'o-ring thigh strap', 'frilled socks', 'swim briefs', 'leotard pull', 'grey necktie', 'breast curtain', 'holstered', 'puffy shorts', 'cherry blossom print', 'helm', 'blue sash', 'two-sided jacket', 'crotch plate', 'plaid necktie', 'kappougi', 'multicolored legwear', 'orange scarf', 'backless leotard', 'multicolored gloves', 'holding swim ring', 'multiple condoms', 'loose clothes', 'horned helmet', 'mechanical tail', 'mini-hakkero', 'stomach cutout', 'naked sheet', 'skull print', 'seigaiha', 'chaps', 'bird on hand', 'poke ball print', 'black male underwear', 'happy tears', 'emoji', 'nijigasaki academy school uniform', 'clothed female nude female', 'tape gag', 'st\. gloriana\'s military uniform', 'constellation print', 'leg belt', 'see-through skirt', 'sports bikini', 'tracen training uniform', 'star necklace', 'fine fabric emphasis', 'open pants', 'fishnet top', 'drop earrings', 'multicolored bikini', 'barcode tattoo', 'sleeve garter', 'heart o-ring', 'pink vest', 'two-tone ribbon', 'bear hair ornament', 'chest strap', 'bowl hat', 'tight shirt', 'brown necktie', 'pencil dress', 'gold hairband', 'yellow hoodie', 'condom packet strip', 'white bag', 'red-tinted eyewear', 'animal ear headwear', 'collared coat', 'volleyball uniform', 'budget sarashi', 'anzio school uniform', 'collared cape', 'grey scarf', 'yellow thighhighs', 'cloud print', 'green sleeves', 'yellow bra', 'fish hair ornament', 'poncho', 'dixie cup hat', 'tankini', 'purple gemstone', 'bondage outfit', 'scar on neck', 'lip piercing', 'checkered kimono', 'clover print', 'ushanka', 'lycoris uniform', 'holding game controller', 'untucked shirt', 'pink socks', 'black buruma', 'mars symbol', 'winged helmet', 'skull earrings', 'side-tie leotard', 'party hat', 'green apron', 'gusset', 'gold necklace', 'mouth veil', 'polka dot dress', 'puffy pants', 'plaid headwear', 'space helmet', 'brown hairband', 'sidepec', 'strawberry print', 'leather belt', 'butler', 'pokemon on head', 'claw ring', 'super robot', 'frilled cuffs', 'two-tone bowtie', 'baseball uniform', 'single gauntlet', 'taut dress', 'holding brush', 'black halo', 'checkered necktie', 'three-dimensional maneuver gear', 'tangzhuang', 'cropped vest', 'utility belt', 'white serafuku', 'fur-trimmed cloak', 'straitjacket', 'blue headband', 'gold bracelet', 'pink coat', 'black undershirt', 'stiletto heels', 'poke ball symbol', 'cum on legs', 'polka dot bra', 'holding baseball bat', 'naked cape', 'orange thighhighs', 'thigh cutout', 'ankle lace-up', 'open bra', 'ribbon-trimmed clothes', 'polo shirt', 'blue bag', 'purple belt', 'strap-on', 'red bandana', 'blue collar', 'gorget', 'white veil', 'belt bra', 'yellow armband', 'holding leaf', 'flower ornament', 'german clothes', 'fur-trimmed skirt', 'shoulder boards', 'flame print', 'cupless bra', 'holding shoes', 'hooded sweater', 'arm wrap', 'multicolored shirt', 'pillbox hat', 'brown socks', 'single fingerless glove', 'plaid pants', 'holding helmet', 'claw \(weapon\)', 'yellow belt', 'pink sailor collar', 'homurahara academy school uniform', 'red hood', 'grey sleeves', 'pocky kiss', 'unworn bikini top', 'striped gloves', 'hair ears', 'bow earrings', 'fur-trimmed kimono', 'cropped hoodie', 'bandaid on hand', 'biker clothes', 'sticker on face', 'pink pajamas', 'green-framed eyewear', 'bandaged neck', 'pacifier', 'striped kimono', 'crescent facial mark', 'x', 'blue cloak', 'stitched face', 'sweatpants', 'shoulder strap', 'eyewear on headwear', 'cowboy western', 'pink collar', 'respirator', 'unworn boots', 'ribbon bondage', 'male playboy bunny', 'thigh belt', 'shoelaces', 'kibito high school uniform', 'purple capelet', 'yellow bag', 'bodice', 'pink eyeshadow', 'holding pillow', 'dress tug', 'pink belt', 'reindeer costume', 'ribbon-trimmed collar', 'hakama pants', 'snap-fit buckle', 'chef', 'pink one-piece swimsuit', 'gold armor', 'magatama earrings', 'holding balloon', 'brown-framed eyewear', 'blue-tinted eyewear', 'moon \(ornament\)', 'buttoned cuffs', 'cow boy', 'micro panties', 'viewer holding leash', 'wiping tears', 'priest', 'purple eyeshadow', 'yellow sash', 'sword over shoulder', 'holding scissors', 'brown cloak', 'multicolored bow', 'romper', 'diamond cutout', 'kuromorimine school uniform', 'single strap', 'shinsengumi', 'single pantsleg', 'bird on shoulder', 'yasogami school uniform', 'gold bikini', 'grey belt', 'black garter straps', 'undone necktie', 'orange sailor collar', 'ankle strap', 'holding needle', 'triangle earrings', 'bow choker', 'striped shorts', 'platform heels', 'delinquent', 'ribbed sleeves', 'animal hug', 'dress flower', 'embellished costume', 'thighhighs pull', 'hooded robe', 'purple-tinted eyewear', 'venus symbol', 'yellow pants', 'heart button', 'sunflower hair ornament', 'hawaiian shirt', 'plate armor', 'bruise on face', 'sleeveless turtleneck leotard', '39', 'plaid jacket', 'lace-trimmed sleeves', 'orange neckerchief', 'pointless condom', 'drinking straw in mouth', 'diving suit', 'dirndl', 'sakuramon', 'holding water gun', 'two-tone headwear', 'brown bowtie', 'ribbon in mouth', 'frilled shorts', 'green bodysuit', 'tricorne', 'handkerchief', 'spiked club', 'cloth gag', 'harem pants', 'naked kimono', 'vibrator under panties', 'leather gloves', 'sleeveless hoodie', 'naked hoodie', 'multicolored coat', 'tribal', 'colored shoe soles', 'bow legwear', 'sparkler', 'mustache stubble', 'greco-roman clothes', 'butterfly on hand', 'turtleneck leotard', 'gradient clothes', 'sleep mask', 'hakurei reimu \(cosplay\)', 'ass cutout', 'latex gloves', 'bath yukata', 'year of the dragon', 'santa boots', 'bear print', 'gold choker', 'open robe', 'drawing bow', 'icho private high school uniform', 'ginkgo leaf', 'scar on stomach', 'loose bowtie', 'grey bikini', 'unworn sandals', 'yellow coat', 'white armor', 'forked tongue', 'eyewear strap', 'print bra', 'pentacle', 'shimaidon \(sex\)', 'blue armor', 'pink pantyhose', 'kigurumi', 'happi', 'duffel coat', 'pants rolled up', 'unworn gloves', 'short jumpsuit', 'grey ribbon', 'volleyball \(object\)', 'deerstalker', 'red apron', 'star facial mark', 'broken chain', 'grey sports bra', 'orange pants', 'tulip hat', 'untying', 'orange pantyhose', 'ajirogasa', 'wrist guards', 'grey bra', 'ballerina', 'full-length zipper', 'novel cover', 'cross print', 'masturbation through clothes', 'black garter belt', 'purple one-piece swimsuit', 'green capelet', 'holding fishing rod', 'two-tone footwear', 'overcoat', 'dark penis', 'key necklace', 'winged footwear', 'brown apron', 'high kick', 'pink-tinted eyewear', 'holding cane', 'crescent print', 'mask around neck', 'brown hoodie', 'print jacket', 'jaguar ears', 'lace-trimmed skirt', 'open belt', 'fishnet gloves', 'naked bandage', 'back-seamed legwear', 'cocktail dress', 'two-tone bodysuit', 'brown bikini', 'torn jeans', 'holding vegetable', 'purple hoodie', 'sunflower field', 'animal ear legwear', 'holding hose', 'new school swimsuit', 'sphere earrings', 'hamaya', 'low neckline', 'yellow apron', 'green bag', 'hatsune miku \(cosplay\)', 'ribbed bodysuit', 'impossible swimsuit', 'cum on self', 'triangle print', 'sunscreen', 'boxer briefs', 'striped sweater', 'candy hair ornament', 'kesa', 'gradient legwear', 'holding jacket', 'mismatched sleeves', 'scooter', 'kimono skirt', 'orange ascot', 'tooth necklace', 'purple neckerchief', 'double fox shadow puppet', 'aqua panties', 'sideless shirt', 'leather boots', 'goatee stubble', 'hand tattoo', 'ballet slippers', 'camouflage jacket', 'kimono pull', 'combat helmet', 'grey neckerchief', 'tapir tail', 'single horizontal stripe', 'white bird', 'glomp', 'diving mask on head', 'gradient dress', 'pointy footwear', 'blood on knife', 'torn scarf', 'kouhaku nawa', 'spiked choker', 'sword on back', 'kiyosumi school uniform', 'holding stylus', 'arrow through heart', 'scar on leg', 'sobu high school uniform', 'onmyouji', 'huge bow', 'nippleless clothes', 'aqua jacket', 'circle skirt', 'sleeve bow', 'no gloves', 'pearl bracelet', 'orange hoodie', 'hooded cardigan', 'pink capelet', 'yellow bodysuit', 'two-tone sports bra', 'combat boots', 'rabbit-shaped pupils', 'yin yang orb', 'dice hair ornament', 'fish print', 'polka dot swimsuit', 'ninja mask', 'overall shorts', 'holding ladle', 'sweaty clothes', 'shorts under dress', 'fur-trimmed footwear', 'shiny legwear', 'drum set', 'eden academy school uniform', 'eyewear hang', 'star brooch', 'kirin \(armor\)', 'expressive clothes', 'burnt clothes', 'ribbon-trimmed dress', 'multicolored headwear', 'duster', 'necktie grab', 'wetsuit', 'cross tie', 'belt boots', 'sharp toenails', 'camouflage pants', 'ribbed leotard', 'torn leotard', 'pinching sleeves', 'strawberry hair ornament', 'food-themed earrings', 'white umbrella', 'holding ice cream', 'torn panties', 'green socks', 'clothes', 'plaid panties', 'mole above mouth', 'riding pokemon', 'athletic leotard', 'headlamp', 'sword behind back', 'grey bodysuit', 'fur-trimmed shorts', 'frilled leotard', 'jingasa', 'brown corset', 'bird mask', 'orange panties', 'hat tip', 'tarot \(medium\)', 'denim jacket', 'two-tone hairband', 'wig', 'square 4koma', 'brown panties', 'holding gohei', 'anchor print', 'white snake', 'polka dot headwear', 'white garter straps', 'frilled ascot', 'colored shadow', 'yellow sleeves', 'age regression', 'shark costume', 'cutout above navel', 'purple scrunchie', 'torn gloves', 'two-tone legwear', 'motorcycle helmet', 'high-waist pantyhose', 'mummy costume', 'orange sweater', 'mahjong tile', 'unitard', 'torn jacket', 'bikesuit', 'upshorts', 'papakha', 'lace-trimmed gloves', 'silver trim', 'scarf over mouth', 'lace choker', 'collared vest', 'tented shirt', 'ghost costume', 'animal on lap', 'ballet', 'penis peek', 'crystal earrings', 'double w', 'bicorne', 'holding saucer', 'multicolored footwear', 'kourindou tengu costume', 'red border', 'pink border', 'detective', 'multicolored kimono', 'drawing sword', 'vampire costume', 'shell bikini', 'brown leotard', 'pink ascot', 'breast cutout', 'two-tone leotard', 'holding violin', 'stole', 'cetacean tail', 'holding envelope', 'sparkle print', 'yellow leotard', 'frog print', 'yellow butterfly', 'pink camisole', 'panties on head', 'lapel pin', 'loungewear', 'nearly naked apron', 'long tail', 'green hakama', 'santa gloves', 'kodona', 'pearl earrings', 'blue border', 'boobplate', 'heart collar', 'training bra', 'arm armor', 'purple socks', 'white mask', 'fourth east high school uniform', 'polka dot legwear', 'uchikake', 'surrounded by penises', 'print headwear', 'pouring onto self', 'egg hair ornament', 'kamiyama high school uniform \(hyouka\)', 'baggy clothes', 'kine', 'yellow cape', 'native american', 'hanten \(clothes\)', 'buruma pull', 'holding bucket', 'adjusting legwear', 'lace gloves', 'side drill', 'sideburns stubble', 'tube dress', 'blue sports bra', 'strap lift', 'scar on mouth', 'nejiri hachimaki', 'gathers', 'covering one eye', 'rook \(chess\)', 'glowing butterfly', 'thighhighs over pantyhose', 'wakizashi', 'swim cap', 'fur cape', 'grey capelet', 'stained panties', 'aviator sunglasses', 'pill earrings', 'blue robe', 'prison clothes', 'aqua footwear', 'drying hair', 'unzipping', 'pinstripe shirt', 'hat over one eye', 'full beard', 'bishop \(chess\)', 'bandaid hair ornament', 'huge moon', 'hanbok', 'loose shirt', 'year of the rat', 'footwear ribbon', 'tearing clothes', 'white butterfly', 'grey hairband', 'ornate ring', 'coin hair ornament', 'holding tablet pc', 'bucket hat', 'gold footwear', 'tutu', 'holding popsicle', 'between pectorals', 'orange vest', 'alpaca ears', 'holding ribbon', 'floating scarf', 'mole on cheek', 'crotch cutout', 'single epaulette', 'heart facial mark', 'cropped sweater', 'messenger bag', 'weasel ears', 'cowboy boots', 'wrist bow', 'upshirt', 'in cup', 'brown sleeves', 'clothes between breasts', 'swimsuit cover-up', 'double vertical stripe', 'covering ass', 'kissing hand', 'armpit cutout', 'white hood', 'brown choker', 'chin strap', 'gladiator sandals', 'mole on stomach', 'single boot', 'red tank top', 'black umbrella', 'blue tunic', 'wrist wrap', 'single wrist cuff', 'kepi', 'white headdress', 'wet dress', 'hooded track jacket', 'orange sleeves', 'brown collar', 'two-tone cape', 'hooded bodysuit', 'red mask', 'body armor', 'red mittens', 'torn swimsuit', 'purple sash', 'satin', 'alice \(alice in wonderland\) \(cosplay\)', 'cat ear legwear', 'saiyan armor', 'white mittens', 'grey cape', 'frilled sailor collar', 'side slit shorts', 'pants tucked in', 'condom belt', 'cross choker', 'black sweater vest', 'rider belt', 'multicolored cape', 'girthy penis', 'yellow socks', 'fold-over boots', 'pink hakama', 'naked overalls', 'spit take', 'leg wrap', 'mochi trail', 'sleeve ribbon', 'blood on arm', 'tied jacket', 'cum in nose', 'blue tank top', 'two-sided dress', 'holding beachball', 'clothes between thighs', 'purple-framed eyewear', 'jockstrap', 'lowleg pants', 'flying kick', 'tight dress', 'no jacket', 'holding jewelry', 'frilled camisole', 'unworn coat', 'see-through jacket', 'pink cape', 'sideways hat', 'holding megaphone', 'string bra', 'huge testicles', 'unworn dress', 'holding letter', 'coke-bottle glasses', 'open bodysuit', 'holding behind back', 'holding chocolate', 'studded bracelet', 'aqua gloves', 'star pasties', 'shuka high school uniform', 'multicolored scarf', 'test plugsuit', 'levitation', 'houndstooth', 'head chain', 'yellow tank top', 'polka dot skirt', 'radiation symbol', 'chalice', 'adidas', 'bandaid on forehead', 'vertical-striped jacket', 'leather pants', 'hard hat', 'cardigan around waist', 'vertical-striped bikini', 'torn bodystocking', 'shoulder cannon', 'purple ascot', 'breast padding', 'white tiger', 'arachne', 'cross pasties', 'holding money', 'two-tone hoodie', 'kimono lift', 'nipple clamps', 'latex legwear', 'grey tank top', 'back-print panties', 'barefoot sandals \(jewelry\)', 'green pantyhose', 'heart maebari', 'male maid', 'arm cuffs', 'floral print kimono', 'fake nails', 'ribbon-trimmed headwear', 'clown', 'joestar birthmark', 'taimanin suit', 'jaguar print', 'adjusting necktie', 'lightsaber', 'jeweled branch of hourai', 'multi-strapped panties', 'medallion', 'holding notebook', 'pool of blood', 'yellow raincoat', 'flower over eye', 'cardigan vest', 'bridal legwear', 'yellow-tinted eyewear', 'striped hoodie', 'naked scarf', 'dudou', 'green tunic', 'otter ears', 'purple hakama', 'green tank top', 'hand under shirt', 'skirt basket', 'white romper', 'sitting backwards', 'youtou high school uniform', 'vietnamese dress', 'lace-trimmed hairband', 'hoodie lift', 'bear costume', 'green belt', 'crotchless pantyhose', 'wringing clothes', 'holding branch', 'shorts around one leg', 'aqua neckerchief', 'holding remote control', 'nose ring', 'jacket pull', 'polka dot shirt', 'underbutt', 'holding skull', 'four-leaf clover hair ornament', 'potara earrings', 'grey leotard', 'print necktie', 'parka', 'shell necklace', 'holding sex toy', 'blue bandana', 'apron lift', 'long beard', 'orange belt', 'animal slippers', 'camouflage headwear', 'penguin hood', 'crocs', 'jacket over swimsuit', 'rope belt', 'polar bear', 'shoulder sash', 'fur boots', 'checkered sash', 'yellow sweater vest', 'purple cardigan', 'anchor necklace', 'striped hairband', 'brown bra', 'sailor senshi', 'bike shorts under shorts', 'hat with ears', 'puff and slash sleeves', 'stitched mouth', 'half mask', 'print sleeves', 'green scrunchie', 'thick mustache', 'argyle sweater', 'hospital gown', 'onesie', 'green armband', 'polka dot scrunchie', 'double \\m/', 'two-tone coat', 'cherry hair ornament', 'sukajan', 'platform boots', 'floating weapon', 'wa lolita', 'striped one-piece swimsuit', 'cat costume', 'jaguar tail', 'rabbit hat', 'thick beard', 'yellow border', 'martial arts belt', 'bib', 'fur-trimmed collar', 'sports bra lift', 'surcoat', 'single thigh boot', 'strawberry panties', 'high tops', 'sitting on table', 'plaid bra', 'sleeve grab', 'panty lift', 'blood bag', 'ankle wrap', 'male underwear pull', 'print hoodie', 'green-tinted eyewear', 'dress swimsuit', 'flower brooch', 'cum in container', 'cross-laced legwear', 'popped button', 'blue shawl', 'butterfly brooch', 'white sarong', 'green one-piece swimsuit', 'grey serafuku', 'lace-trimmed thighhighs', 'orange cape', 'american flag print', 'skirt flip', 'ehoumaki', 'chain headband', 'holding frying pan', 'orange leotard', 'sling bikini top', 'adapted uniform', 'kabuto \(helmet\)', 'planet hair ornament', 'hair color connection', 'patchwork clothes', 'hat on back', 'watermelon slice', 'holding teapot', 'pants under skirt', 'unworn bikini bottom', 'popsicle in mouth', 'milky way', 'multicolored hairband', 'drop-shaped pupils', 'skull necklace', 'purple serafuku', 'mitre', 'frilled jacket', 'penis on ass', 'aqua bra', 'blue pajamas', 'anchor choker', 'polka dot ribbon', 'halter shirt', 'red sports bra', 'nudist', 'naked tabard', 'sideless kimono', 'single knee pad', 'long shirt', 'multiple scars', 'penis in panties', 'cross-laced slit', 'card parody', 'orange socks', 'cream on face', 'sam browne belt', 'satin panties', 'embroidery', 'blue sarong', 'pink umbrella', 'buruma aside', 'genderswap \(otf\)', 'blue umbrella', 'legband', 'musical note print', 'holding wrench', 'unworn eyepatch', 'hooded dress', 'floating book', 'rabbit costume', 'skeleton print', 'wataboushi', 'st\. theresa\'s girls academy school uniform', 'pinstripe suit', 'bowler hat', 'pegasus knight uniform \(fire emblem\)', 'green eyeshadow', 'pumpkin hair ornament', 'bandaged wrist', 'holding swimsuit', 'spiked hairband', 'coat dress', 'jester', 'stopwatch', 'shoulder belt', 'holding footwear', 'holding toy', 'panties under buruma', 'food art', 'hugging book', 'brown border', 'half-skirt', 'orange jumpsuit', 'midriff sarashi', 'red track suit', 'grey suit', 'hooded vest', 'scylla', 'bathrobe', 'coif', 'bikini shorts', 'bow skirt', 'side-tie peek', 'tweaking own nipple', 'bralines', 'blue camisole', 'striped coat', 'pelt', 'unfastened', 'greek toe', 'black armband', 'adjusting panties', 'vertical-striped socks', 'plaid ribbon', 'vertical-striped panties', 'print sarong', 'cloth', 'holding test tube', 'band uniform', 'checkered shirt', 'lowleg skirt', 'fur-trimmed shirt', 'german flag bikini', 'lightning bolt print', 'holding mop', 'blue tabard', 'holly hair ornament', 'exercise ball', 'lillian girls\' academy school uniform', 'covering one breast', 'vertical-striped pants', 'blood on leg', 'stained clothes', 'high-low skirt', 'christmas stocking', 'tengu mask', 'pumpkin hat', 'hand wraps', 'belt skirt', 'silver dress', 'lace-trimmed choker', 'brown mittens', 'shiny and normal', 'blue hood', 'naked cloak', 'one-piece thong', 'black bandeau', 'orange goggles', 'fishnet socks', 'purple collar', 'flower choker', 'elbow sleeve', 'holding heart', 'pocky in mouth', 'grey apron', 'jiangshi costume', 'mizu happi', 'rubber gloves', 'red cardigan', 'holding coin', 'mole under each eye', 'clothes theft', 'simulated fellatio', 'holding microphone stand', 'clock eyes', 'holding chain', 'wrong foot', 'converse', 'thong aside', 'walking on liquid', 'knight \(chess\)', 'pelvic curtain lift', 'mutual hug', 'brown neckerchief', 'kerchief', 'red suit', 'red robe', 'strapless bottom', 'wing brooch', 'diagonal-striped bowtie', 'holding drumsticks', 'aqua kimono', 'vertical-striped kimono', 'stitched arm', 'pink sash', 'cuff links', 'checkered dress', 'ornate border', 'animal ear hairband', 'grey bowtie', 'clothed male nude male', 'toe cleavage', 'yellow camisole', 'crotch zipper', 'shirt overhang', 'animal on hand', 'holding shirt', 'unworn shorts', 'riding bicycle', 'star-shaped eyewear', 'orange headband', 'scouter', 'long toenails', 'holding cake', 'cargo pants', 'frilled umbrella', 'glitter', 'holding suitcase', 'green headband', 'micro bra', 'motosu school uniform', 'brown serafuku', 'single head wing', 'year of the dog', 'covered clitoris', 'panda hood', 'taut swimsuit', 'purple butterfly', 'aqua leotard', 'little red riding hood \(grimm\) \(cosplay\)', 'year of the pig', 'fur cuffs', 'glowing hand', 'panties under shorts', 'maple leaf print', 'exploding clothes', 'right-over-left kimono', 'holding creature', 'stiletto \(weapon\)', 'sock pull', 'clawed gauntlets', 'print mug', 'camisole lift', 'frilled headwear', 'cable tail', 'red male underwear', 'exposed pocket', 'two-sided coat', 'safety glasses', 'holding fish', 'front slit', 'flippers', 'kariyushi shirt', 'knives between fingers', 'broken sword', 'policeman', 'spade hair ornament', 'male underwear peek', 'leotard peek', 'neck garter', 'weasel tail', 'blue suit', 'holding photo', 'dissolving clothes', 'holding pole', 'jacket lift', 'holding shovel', 'backless swimsuit', 'tickling armpits', 'low-cut armhole', 'propeller hair ornament', 'fake magazine cover', 'holding cross', 'otter tail', 'taut leotard', 'o-ring swimsuit', 'wind turbine', 'pom pom earrings', 'checkered bow', 'multiple hairpins', 'studded choker', 'red bandeau', 'single garter strap', 'fruit hat ornament', 'ski goggles', 'holding briefcase', 'brown sash', 'layered kimono', 'o-ring belt', 'striped vest', 'green cardigan', 'multicolored stripes', 'aqua hairband', 'plate carrier', 'bear hood', 'holding bra', 'detached leggings', 'paw print pattern', 'body switch', 'multicolored tail', 'walker \(robot\)', 'down jacket', 'rabbit on head', 'giant male', 'holding scroll', 'pink tank top', 'yellow one-piece swimsuit', 'white bandeau', 'black tube top', 'scoop neck', 'female goblin', 'temari ball', 'red wine', 'yellow pantyhose', 'bandaged fingers', 'ahoge wag', 'black hood', 'black veil', 'head on head', 'leaf background', 'hakama shorts', 'moose ears', 'fishnet bodysuit', 'pointy hat', 'fur jacket', 'bandaid on neck', 'holding surfboard', 'bridal lingerie', 'hat belt', 'overall skirt', 'sweater pull', 'holding map', 'disguise', 'knife sheath', 'rotary phone', 'pantyhose under swimsuit', 'pawn \(chess\)', 'unworn goggles', 'sky lantern', 'frontless outfit', 'armored leotard', 'shoulder plates', 'ribbed thighhighs', 'forked tail', 'lightning bolt hair ornament', 'undone neck ribbon', 'shoulder guard', 'lop rabbit ears', 'cassock', 'metamoran vest', 'normal suit', 'checkered legwear', 'see-through swimsuit', 'holding necklace', 'panties over pantyhose', 'orange bra', 'adjusting scarf', 'layered shirt', 'bird on arm', 'paint on clothes', 'scar on hand', 'blue outline', 'unworn bikini', 'pink sports bra', 'tape on nipples', 'adjusting buruma', 'side-tie shirt', 'torn coat', 'rash guard', 'poke ball \(legends\)', 'ankle bow', 'covering own ears', 'mtu virus', 'bandaid on head', 'fur-trimmed bikini', 'hat tassel', 'argyle cutout', 'cross-laced skirt', 'fruit on head', 'suspenders slip', 'cow costume', 'multicolored leotard', 'white garter belt', 'holding toothbrush', 'toga', 'holding lipstick tube', 'multi-strapped bikini top', 'white wristband', 'purple robe', 'turtleneck jacket', 'rice hat', 'shared earphones', 'mole on arm', 'holding mirror', 'corsage', 'black outline', 'anchor earrings', 'wrapped candy', 'gingham', 'sweet lolita', 'side-tie skirt', 'print scarf', 'green collar', 'sweater tucked in', 'front-print panties', 'square neckline', 'bear panties', 'mini witch hat', 'holding key', 'holding torch', 'holding plectrum', 'white tube top', 'unworn hair ornament', 'holding magnifying glass', 'single off shoulder', 'torn cloak', 'heart hair', 'shirt around waist', 'sailor swimsuit \(idolmaster\)', 'detached ahoge', 'ankle garter', 'year of the rooster', 'singlet', 'sailor collar lift', 'aviator cap', 'aqua shorts', 'holding newspaper', 'female service cap', 'ankleband', 'black babydoll', 'multiple bracelets', 'front zipper swimsuit', 'kin-iro mosaic high school uniform', 'holding bell', 'blue male underwear', 'side cape', 'glove bow', 'green serafuku', 'claw foot bathtub', 'ribbed socks', 'dress shoes', 'vertical-striped shorts', 'blue sweater vest', 'fur-trimmed thighhighs', 'streetwear', 'vertical stripes', 'labcoat', 'argyle', 'print legwear', 'tight', 'legwear under shorts', 'skirt removed', 'panties removed', 'multi-strapped bikini', 'diagonal stripes', 'clothes removed', 'bikini lift', 'gothic', 'frilled swimsuit', 'bunny print', 'qing guanmao', 'matching outfit', 'borrowed garments', 'beltbra', 'bikini top removed', 'nike', 'traditional clothes', 'boots removed', 'power suit \(metroid\)', 'sandals removed', 'clog sandals', 'multiple straps', 'socks removed', 'catholic', 'barefoot sandals', 'dress removed', 'strapless swimsuit', 'sling', 'bunny hat', 'beltskirt', 'greek clothes', 'military helmet', 'hardhat', 'bikini bottom removed', 'yamakasa', 'necktie removed'
}

nudity_tags = {
    'completely nude', 'nude', 'no pants', 'no bra', 'no panties', 'no shirt', 'topless', 'bottomless', 'underwear only', 'breasts out', 'areola slip', 'nipple slip', 'nipples', 'midriff', 'navel', 'anus', 'nipples', 'pussy', 'penis', 'ass', 'breasts', 'cleavage', 'swimsuit', 'thighs'
}

ornament_tags = {
    'hair ornament', 'hat', 'bow', 'ribbon', 'jewelry', 'hair ribbon', 'hair bow', 'earrings', 'frills', 'hairband', 'choker', 'hairclip', 'bowtie', 'hood', 'hair flower', 'necklace', 'halo', 'red ribbon', 'black headwear', 'black choker', 'white headwear', 'blue bow', 'witch hat', 'blush stickers', 'headgear', 'black hairband', 'eyepatch', 'scrunchie', 'white bow', 'mob cap', 'helmet', 'feathers', 'x hair ornament', 'blue ribbon', 'cross', 'hat ribbon', 'crown', 'pink bow', 'ear piercing', 'gem', 'red bowtie', 'lipstick', 'goggles', 'eyewear on head', 'hooded jacket', 'red headwear', 'white ribbon', 'ribbon trim', 'baseball cap', 'blue headwear', 'red neckerchief', 'yellow bow', 'neck bell', 'tokin hat', 'green bow', 'hair scrunchie', 'santa hat', 'pink ribbon', 'single earring', 'headset', 'beads', 'unworn headwear', 'purple bow', 'goggles on head', 'black bowtie', 'hat ornament', 'yellow ribbon', 'top hat', 'sun hat', 'white hairband', 'star hair ornament', 'circlet', 'crystal', 'hairpin', 'hoop earrings', 'unworn hat', 'blue bowtie', 'green headwear', 'mini hat', 'military hat', 'brown headwear', 'bespectacled', 'striped bow', 'tress ribbon', 'animal hood', 'heart hair ornament', 'red hairband', 'hair bell', 'bandaid on face', 'nurse cap', 'purple ribbon', 'butterfly hair ornament', 'hat flower', 'hair tie', 'straw hat', 'green ribbon', 'visor cap', 'orange bow', 'stud earrings', 'blindfold', 'headpiece', 'gag', 'beanie', 'eighth note', 'sailor hat', 'cabbie hat', 'frog hair ornament', 'food-themed hair ornament', 'pink headwear', 'hair stick', 'headdress', 'feather hair ornament', 'mask on head', 'horn ornament', 'mini crown', 'blue hairband', 'red choker', 'blue halo', 'earmuffs', 'purple headwear', 'flat cap', 'triangular headpiece', 'snake hair ornament', 'mechanical halo', 'animal hat', 'pink halo', 'white choker', 'lolita hairband', 'facepaint', 'animal nose', 'star earrings', 'crescent hair ornament', 'ribbon choker', 'tate eboshi', 'hat feather', 'garrison cap', 'head wreath', 'frilled bow', 'tilted headwear', 'animal on head', 'ear ornament', 'asymmetrical wings', 'forehead jewel', 'stitches', 'headphones around neck', 'cross necklace', 'crescent hat ornament', 'leaf hair ornament', 'rabbit hair ornament', 'spiked collar', 'neck ring', 'pink bowtie', 'leaf on head', 'polka dot bow', 'heart earrings', 'frilled hairband', 'bonnet', 'earphones', 'tassel earrings', 'star hat ornament', 'grey headwear', 'mini top hat', 'arm ribbon', 'red halo', 'bow hairband', 'yellow halo', 'traditional bowtie', 'anchor hair ornament', 'yellow hairband', 'ear bow', 'ball gag', 'topknot', 'jester cap', 'monocle', 'black blindfold', 'pink choker', 'cube hair ornament', 'cross hair ornament', 'aqua bow', 'pink hairband', 'bandaid on nose', 'skull hair ornament', 'side braids', 'cross earrings', 'horn ribbon', 'plaid bow', 'yellow headwear', 'belt collar', 'brown bow', 'blue butterfly', 'cowboy hat', 'gradient eyes', 'unmoving pattern', 'pearl necklace', 'large bow', 'bird ears', 'fur hat', 'fur-trimmed headwear', 'interface headset', 'frilled hat', 'striped ribbon', 'waist bow', 'fedora', 'cat hair ornament', 'multiple hair bows', 'red scrunchie', 'metal collar', 'hairpods', 'purple hairband', 'wrist ribbon', 'pirate hat', 'orange bowtie', 'orange headwear', 'carrot hair ornament', 'heart necklace', 'backwards hat', 'red headband', 'earclip', 'earpiece', 'pince-nez', 'musical note hair ornament', 'heart choker', 'orange ribbon', 'circle', 'hair beads', 'white feathers', 'dangle earrings', 'cat hood', 'blue scrunchie', 'christmas ornaments', 'bandaid on cheek', 'wizard hat', 'cat ear headphones', 'bandage over one eye', 'wing hair ornament', 'bone hair ornament', 'medical eyepatch', 'braided hair rings', 'multicolored wings', 'purple wings', 'chain necklace', 'ear ribbon', 'black headband', 'mole on thigh', 'multiple earrings', 'bat hair ornament', 'goggles on headwear', 'white scrunchie', 'red eyeliner', 'wiffle gag', 'black scrunchie', 'white headband', 'trefoil', 'horn bow', 'gas mask', 'green hairband', 'horseshoe ornament', 'chef hat', 'earbuds', 'qingdai guanmao', 'slime girl', 'shark hair ornament', 'gold earrings', 'tassel hair ornament', 'triangle', 'orange hairband', 'ankle ribbon', 'flower earrings', 'crescent earrings', 'pink scrunchie', 'strap between breasts', 'winged hat', 'eye mask', 'porkpie hat', 'police hat', 'diagonal-striped bow', 'snowflake hair ornament', 'head-mounted display', 'yellow scrunchie', 'brown ribbon', 'whistle around neck', 'heart in mouth', 'bandaged head', 'black feathers', 'o-ring choker', 'clover hair ornament', 'chinese knot', 'frilled ribbon', 'enpera', 'feather earrings', 'laurel crown', 'large hat', 'pom pom hair ornament', 'ear tag', 'grey bow', 'single ear cover', 'hexagram', 'crossed bandaids', 'bit gag', 'orange scrunchie', 'aqua ribbon', 'smiley face', 'character hair ornament', 'star halo', 'cat hat', 'shako cap', 'diadem', 'magatama necklace', 'star choker', 'bandage on face', 'prayer beads', 'd-pad hair ornament', 'school hat', 'helm', 'plaid necktie', 'horned helmet', 'seigaiha', 'emoji', 'star necklace', 'drop earrings', 'two-tone ribbon', 'bear hair ornament', 'bowl hat', 'gold hairband', 'animal ear headwear', 'fish hair ornament', 'dixie cup hat', 'lip piercing', 'ushanka', 'skull earrings', 'party hat', 'plaid headwear', 'brown hairband', 'black halo', 'blue headband', 'flower ornament', 'pillbox hat', 'hair ears', 'bow earrings', 'sticker on face', 'pacifier', 'stitched face', 'respirator', 'pink eyeshadow', 'magatama earrings', 'moon \(ornament\)', 'cow boy', 'purple eyeshadow', 'multicolored bow', 'triangle earrings', 'sunflower hair ornament', 'bruise on face', 'sakuramon', 'two-tone headwear', 'drawing bow', 'grey ribbon', 'tulip hat', 'jaguar ears', 'sphere earrings', 'candy hair ornament', 'combat helmet', 'diving mask on head', 'spiked choker', 'sword on back', 'arrow through heart', 'huge bow', 'sleeve bow', 'dice hair ornament', 'multicolored headwear', 'strawberry hair ornament', 'food-themed earrings', 'two-tone hairband', 'wig', 'polka dot headwear', 'purple scrunchie', 'crystal earrings', 'panties on head', 'pearl earrings', 'egg hair ornament', 'side drill', 'swim cap', 'pill earrings', 'hat over one eye', 'full beard', 'bandaid hair ornament', 'footwear ribbon', 'grey hairband', 'coin hair ornament', 'bucket hat', 'alpaca ears', 'weasel ears', 'wrist bow', 'white headdress', 'sleeve ribbon', 'sideways hat', 'head chain', 'bandaid on forehead', 'hard hat', 'ribbon-trimmed headwear', 'flower over eye', 'otter ears', 'lace-trimmed hairband', 'four-leaf clover hair ornament', 'potara earrings', 'long beard', 'camouflage headwear', 'striped hairband', 'hat with ears', 'green scrunchie', 'thick mustache', 'polka dot scrunchie', 'cherry hair ornament', 'rabbit hat', 'thick beard', 'chain headband', 'planet hair ornament', 'multicolored hairband', 'polka dot ribbon', 'unworn eyepatch', 'bowler hat', 'green eyeshadow', 'pumpkin hair ornament', 'spiked hairband', 'plaid ribbon', 'holly hair ornament', 'pumpkin hat', 'mole under each eye', 'clock eyes', 'animal ear hairband', 'orange headband', 'green headband', 'single head wing', 'frilled headwear', 'spade hair ornament', 'propeller hair ornament', 'pom pom earrings', 'checkered bow', 'fruit hat ornament', 'aqua hairband', 'head on head', 'moose ears', 'pointy hat', 'lightning bolt hair ornament', 'undone neck ribbon', 'lop rabbit ears', 'ankle bow', 'bandaid on head', 'hat tassel', 'fruit on head', 'rice hat', 'anchor earrings', 'mini witch hat', 'unworn hair ornament', 'heart hair', 'detached ahoge', 'aviator cap', 'female service cap', 'glove bow', 'bunny hat'
}

holding_tags = {
    'holding', 'holding weapon', 'holding hands', 'holding sword', 'holding food', 'holding gun', 'holding cup', 'holding phone', 'holding book', 'holding umbrella', 'holding staff', 'holding clothes', 'holding flower', 'holding knife', 'holding bag', 'holding bottle', 'holding poke ball', 'holding tray', 'holding fan', 'holding polearm', 'holding microphone', 'holding animal', 'holding instrument', 'holding hair', 'holding bouquet', 'holding gift', 'holding stuffed toy', 'holding another\'s wrist', 'holding plate', 'holding fruit', 'holding chopsticks', 'holding spoon', 'holding bow \(weapon\)', 'holding fork', 'holding card', 'holding can', 'holding cigarette', 'holding wand', 'holding hat', 'holding candy', 'holding paper', 'holding ball', 'holding removed eyewear', 'holding pen', 'holding camera', 'holding strap', 'holding broom', 'holding shield', 'holding bowl', 'holding lollipop', 'holding towel', 'holding own arm', 'holding box', 'holding mask', 'holding scythe', 'holding pom poms', 'holding axe', 'holding pokemon', 'holding another\'s arm', 'holding dagger', 'holding underwear', 'holding drink', 'holding smoking pipe', 'holding flag', 'holding arrow', 'holding controller', 'holding doll', 'holding sheath', 'holding basket', 'holding panties', 'holding leash', 'holding hammer', 'holding condom', 'holding sign', 'holding sack', 'holding cat', 'holding paintbrush', 'holding handheld game console', 'holding clipboard', 'holding syringe', 'holding pencil', 'holding lantern', 'holding stick', 'holding whip', 'holding swim ring', 'holding game controller', 'holding brush', 'holding baseball bat', 'holding leaf', 'holding shoes', 'holding helmet', 'holding pillow', 'holding balloon', 'viewer holding leash', 'holding scissors', 'holding needle', 'holding water gun', 'holding fishing rod', 'holding cane', 'holding vegetable', 'holding hose', 'holding jacket', 'holding own tail', 'holding stylus', 'holding ladle', 'holding ice cream', 'holding gohei', 'holding another\'s hair', 'holding saucer', 'holding violin', 'holding envelope', 'holding legs', 'holding bucket', 'holding pizza', 'holding tablet pc', 'holding popsicle', 'holding ribbon', 'holding beachball', 'holding jewelry', 'holding megaphone', 'holding letter', 'holding behind back', 'holding chocolate', 'holding head', 'holding money', 'holding notebook', 'holding branch', 'holding remote control', 'holding skull', 'holding sex toy', 'holding breath', 'holding frying pan', 'holding teapot', 'holding wrench', 'holding another\'s leg', 'holding riding crop', 'holding swimsuit', 'holding footwear', 'holding toy', 'holding fireworks', 'holding with feet', 'holding test tube', 'holding mop', 'holding heart', 'holding coin', 'holding microphone stand', 'holding chain', 'holding own foot', 'holding drumsticks', 'holding shirt', 'holding cake', 'holding suitcase', 'holding leg', 'holding creature', 'holding fish', 'holding photo', 'holding pole', 'holding shovel', 'holding cross', 'holding briefcase', 'holding bra', 'holding another\'s foot', 'holding scroll', 'holding surfboard', 'holding map', 'holding necklace', 'holding toothbrush', 'holding lipstick tube', 'holding with tail', 'holding mirror', 'holding key', 'holding torch', 'holding plectrum', 'holding magnifying glass', 'holding newspaper', 'holding bell', 'holding eyewear', 'holding innertube'
}

not_scene_tags = appearance_tags.union(clothing_tags)
keep_tags = nudity_tags.union(ornament_tags).union(holding_tags)

patterns_to_keep = [
    r'^.*from_.*$', r'^.*focus.*$', r'^anime.*$', r'^monochrome$', r'^.*background$', r'^comic$', 
    r'^.*censor.*$', r'^.*_name$', r'^signature$', r'^.*_username$', r'^.*text.*$', 
    r'^.*_bubble$', r'^multiple_views$', r'^.*blurry.*$', r'^.*koma$', r'^watermark$', 
    r'^traditional_media$', r'^parody$', r'^.*cover$', r'^.*_theme$', r'^realistic$', 
    r'^oekaki$', r'^3d$', r'^.*chart$', r'^letterboxed$', r'^variations$', r'^.*mosaic.*$', 
    r'^omake$', r'^column.*$', r'^.*_(medium)$', r'^manga$', r'^lineart$', r'^.*logo$', r'^greyscale$',
    r'^.*photorealistic.*$', r'^tegaki$', r'^sketch$', r'^silhouette$', r'^web_address$', r'^.*border$',
    r'^.*photo.*$', r'^.*full_body.*$'
]

clip_adj= [
    'friendly', 'charismatic', 'honest', 'calm', 'independent', 'optimistic', 'generous', 'lively', 'disciplined', 'compassionate', 'hardworking', 'innovative', 'ambitious', 'bold', 'creative', 'outgoing', 'humble', 'selfless', 'practical', 'enthusiastic', 'dependable', 'reliable', 'easygoing', 'assertive', 'responsible', 'considerate', 'cheerful', 'rational', 'analytical', 'insightful', 'open-minded', 'extroverted', 'intelligent', 'confident', 'amiable', 'flexible', 'conscientious', 'authentic', 'fair', 'self-confident', 'skilled', 'gracious', 'diligent', 'positive', 'charming', 'resourceful', 'professional', 'passionate', 'coherent', 'logical', 'empathetic', 'curious', 'immature', 'candid', 'patient', 'genuine', 'kind', 'loyal', 'persistent', 'athletic', 'brave', 'average', 'sociable', 'decisive', 'determined', 'adaptable', 'talented', 'energetic', 'understanding', 'forgiving', 'perceptive', 'tolerant', 'versatile', 'caring', 'fearless', 'trustworthy', 'persevering', 'consistent', 'witty', 'persuasive', 'sensational', 'engaging', 'astute', 'self-disciplined', 'sincere', 'thoughtful', 'wise', 'active', 'adventurous', 'diplomatic', 'gregarious', 'impolite', 'imaginative', 'discreet', 'circumspect', 'neat', 'polite', 'mature', 'sympathetic', 'motivated', 'popular', 'lucky', 'loving', 'nice', 'gentle', 'posh', 'secure', 'good', 'helpful', 'funny', 'intuitive', 'willing', 'powerful', 'realistic', 'inspiring', 'plucky', 'affable', 'communicative', 'composed', 'dynamic', 'amusing', 'meticulous', 'aware', 'careful', 'amicable', 'sassy', 'courteous', 'courageous', 'sappy', 'sardonic', 'faithful', 'humorous', 'bright', 'antisocial', 'annoying', 'shameful', 'belligerent', 'tidy', 'inventive', 'smart', 'joyful', 'antsy', 'sensible', 'romantic', 'sheepish', 'abrasive', 'hopeful', 'complex', 'shameless', 'impressionable', 'irreverent', 'forceful', 'businesslike', 'idiosyncratic', 'intellectual', 'adversarial', 'rebellious', 'silly', 'surprising', 'political', 'outspoken', 'sarcastic', 'unyielding', 'quiet', 'inhibited', 'enigmatic', 'cerebral', 'childlike', 'ethical', 'competitive', 'noncompetitive', 'opportunistic', 'decisive', 'modern', 'shy', 'philosophical', 'mischievous', 'basic', 'introverted', 'relaxed', 'moralistic', 'perfectionist', 'exuberant', 'martyr', 'folksy', 'solemn', 'neutral', 'sensitive', 'straightforward', 'noncommittal', 'unique', 'extravagant', 'casual', 'cultured', 'breezy', 'emotional', 'impartial', 'private', 'unchanging', 'sentimental', 'frugal', 'barbarous', 'soft', 'subservient', 'tough', 'dignified', 'undemanding', 'cautious', 'conservative', 'frank', 'compatible', 'intense', 'mercurial', 'modest', 'ordinary', 'predictable', 'questioning', 'uncompromising', 'reserved', 'timid', 'serious', 'unassuming', 'strict', 'aggressive', 'confrontational', 'devious', 'cocky', 'ignorant', 'forgetful', 'gossipy', 'irrational', 'gullible', 'judgmental', 'pessimistic', 'sleazy', 'prejudiced', 'unmotivated', 'mean', 'argumentative', 'cruel', 'defensive', 'morbid', 'resentful', 'stingy', 'rude', 'obsessive', 'impractical', 'disloyal', 'apologizing', 'disrespectful', 'bossy', 'catty', 'egotistical', 'awkward', 'deceitful', 'dishonest', 'bad', 'evil', 'flaky', 'humorless', 'manipulative', 'nosy', 'petty', 'reckless', 'stupid', 'untidy', 'apathetic', 'boring', 'callous', 'childish', 'cowardly', 'cynical', 'dense', 'dim', 'disorganized', 'disruptive', 'evasive', 'fanatical', 'foolish', 'frivolous', 'greedy', 'grumpy', 'hostile', 'impatient', 'inconsiderate', 'jealous', 'lazy', 'moody', 'nasty', 'stubborn', 'paranoid', 'possessive', 'pretentious', 'rotten', 'selfish', 'spoiled', 'unlucky', 'unreliable', 'vain', 'happy', 'sad', 'angry', 'anxious', 'frustrated', 'sorry', 'ashamed', 'frightened', 'disappointed', 'confused', 'lonely', 'afraid', 'hot', 'crazy', 'withdrawn', 'depressed', 'guilty', 'proud', 'hungry', 'scared', 'excited', 'content', 'bored', 'embarrassed', 'interested', 'thirsty', 'puzzled', 'hurt', 'smug', 'suspicious', 'touching', 'fearful', 'surprised', 'envious', 'nostalgic', 'amused', 'grateful', 'loved', 'miserable', 'regretful', 'sick', 'indifferent', 'enraged', 'pained', 'relieved', 'disgusted', 'ecstatic', 'nervous', 'shocked', 'sleepy', 'skeptical', 'worried', 'attractive', 'dashing', 'fashionable', 'stunning', 'exquisite', 'stylish', 'aesthetic', 'gorgeous', 'trendy', 'lovely', 'handsome', 'radiant', 'beautiful', 'cute', 'pretty', 'alluring', 'striking', 'sophisticated', 'sumptuous', 'elegant', 'classy', 'petite', 'elderly', 'plump', 'thin', 'fat', 'slim', 'well-built', 'stocky', 'slender', 'short', 'chubby', 'stunted', 'ornate', 'stout', 'centenarian', 'youthful', 'spry', 'tall', 'aged', 'venerable', 'ageless', 'muscular', 'lanky', 'octogenarian', 'medium-height', 'stumpy', 'long-lived', 'curvy', 'skinny', 'square', 'rectangular', 'circular', 'round', 'oval', 'flat', 'triangular', 'elliptical', 'spherical', 'symmetrical', 'asymmetrical', 'angular', 'linear', 'humongous', 'conical', 'irregular', 'geometric', 'convex', 'wavy', 'curved', 'twisted', 'cylindrical', 'pyramidal', 'bent', 'blunt', 'straight', 'wide', 'colossal', 'jagged', 'thick', 'substantial', 'narrow', 'microscopic', 'massive', 'oversized', 'snug', 'undersized', 'compact', 'bulky', 'pointed', 'concave', 'gigantic', 'tiny', 'coarse', 'meager', 'stubby', 'roomy', 'fine', 'minuscule', 'large', 'spiky', 'wet', 'bumpy', 'numb', 'blue', 'red', 'black', 'white', 'green', 'pink', 'purple', 'yellow', 'orange', 'brown', 'gray', 'silver', 'lavender', 'turquoise', 'golden', 'magenta', 'teal', 'violet', 'mauve', 'plum', 'lilac', 'indigo', 'bronze', 'cyan', 'navy blue', 'burgundy', 'cherry', 'crimson', 'maroon', 'scarlet', 'cream', 'chartreuse', 'coral', 'salmon', 'olive', 'emerald', 'beige', 'ivory', 'eggshell', 'bone', 'amazing', 'superb', 'exceptional', 'impeccable', 'wonderful', 'supreme', 'phenomenal', 'fabulous', 'astonishing', 'polished', 'magnificent', 'terrific', 'flawless', 'marvelous', 'premium', 'spectacular', 'enchanting', 'stable', 'splendid', 'outstanding', 'majestic', 'incredible', 'heavenly', 'comfortable', 'extraordinary', 'excellent', 'delightful', 'accessible', 'accurate', 'enjoyable', 'frequent', 'occasional', 'constant', 'recurring', 'sporadic', 'continuous', 'repeated', 'regular', 'periodic', 'infrequent', 'intermittent', 'prompt', 'chronic', 'rare', 'young', 'old', 'future', 'past', 'new', 'ancient', 'timeless', 'contemporary', 'current', 'historic', 'vintage', 'antique', 'spicy', 'delicious', 'sweet', 'salty', 'savory', 'nutty', 'tart', 'yummy', 'bland', 'fluffy', 'zesty', 'peppery', 'sinful', 'refreshing', 'crispy', 'tacit', 'buttery', 'decadent', 'robust', 'eggy', 'crumbly', 'juicy', 'bitter', 'piquant', 'astringent', 'delectable', 'high', 'fishy', 'citrusy', 'flavorful', 'saccharine', 'delicate', 'comforting', 'squeaky', 'creamy', 'smoky', 'moist', 'glazed', 'tasty', 'chunky', 'silky', 'pickled', 'fruity', 'fiery', 'honeyed', 'gummy', 'acidic', 'sour', 'refined', 'sizzling', 'rich', 'tangy', 'earthy', 'airy', 'runny', 'velvety', 'melty', 'doughy', 'crunchy', 'leathery', 'chewy', 'mild', 'loud', 'musical', 'melodic', 'roaring', 'dissonant', 'buzzing', 'noisy', 'riotous', 'harmonious', 'raucous', 'whistling', 'mellow', 'humming', 'muted', 'thundering', 'noiseless', 'peaceful', 'rowdy', 'resounding', 'penetrating', 'silent', 'voiceless', 'husky', 'dull', 'jarring', 'subtle', 'muffled', 'speechless', 'percussive', 'pleasing', 'tumultuous', 'deafening', 'gruff', 'rhythmic', 'vocal', 'soundless', 'tranquil', 'glaring', 'unspoken', 'brassy', 'crackling', 'thunderous', 'mute', 'mellifluous', 'vociferous', 'boisterous', 'resonant', 'howling', 'strident', 'screaming', 'tuneful', 'uproarious', 'howling', 'faint', 'rasping', 'croaky', 'raucous', 'discordant', 'piercing', 'shrill', 'screechy', 'insistent', 'pulsating', 'piercing', 'sharp', 'screeching', 'voiced', 'unvoiced', 'clamorous', 'echoing', 'grating', 'audible', 'inaudible', 'grinding', 'monotonous', 'smooth', 'rough', 'shiny', 'synthetic', 'fabric', 'plastic', 'durable', 'heavy', 'ceramic', 'fragile', 'soft', 'porous', 'non-porous', 'brittle', 'matte', 'sticky', 'textured', 'supple', 'lightweight', 'elastic', 'leather', 'natural', 'resilient', 'sturdy', 'opaque', 'spongy', 'tender', 'glass', 'wooden', 'damp', 'warm', 'dry', 'slimy', 'firm', 'cold', 'slick', 'fragrant', 'aromatic', 'antiseptic', 'floral', 'acrid', 'clean', 'rancid', 'foul', 'fetid', 'perfumed', 'evocative', 'funky', 'bouquet', 'pungent', 'frowsty', 'musky', 'scented', 'fusty', 'malodorous', 'musty', 'stuffy', 'overpowering', 'herbal', 'woody', 'flowery', 'nauseating', 'putrid', 'minty', 'lemony', 'sickly', 'peachy', 'coppery', 'garlicky', 'stale', 'ambrosial', 'sunny', 'windy', 'chilly', 'overcast', 'showery', 'icy', 'cloudy', 'clear', 'rainy', 'humid', 'stormy', 'bleak', 'snowy', 'dreary', 'scorching', 'freezing', 'inclement', 'hazy', 'tropical', 'misty', 'foggy', 'clement', 'muggy', 'frosty', 'gloomy', 'scalding', 'cloudless', 'blistering', 'balmy', 'biting', 'brisk', 'temperate', 'windless', 'gusty', 'murky', 'threatening', 'torrential', 'northern', 'southern', 'eastern', 'western', 'northwestern', 'northeastern', 'southwestern', 'southeastern', 'far', 'close', 'near', 'distant', 'inaccessible', 'remote', 'rural', 'suburban', 'urban', 'central', 'peripheral', 'coastal', 'inland', 'galactic', 'adjacent', 'isolated', 'landlocked', 'orbital', 'untouchable', 'continental', 'intercontinental', 'cosmic', 'strong', 'fierce', 'overwhelming', 'dramatic', 'concentrated', 'entertaining', 'fast', 'harsh', 'mighty', 'custom', 'explosive', 'informative', 'electric', 'relevant', 'transparent', 'ferocious', 'vigorous', 'severe', 'convenient', 'potent', 'recyclable', 'maintainable', 'extreme', 'quick', 'concise', 'expert', 'graceful', 'sluggish', 'lavish', 'simple', 'agile', 'clumsy', 'stiff', 'slow', 'jerky', 'regenerative', 'nimble', 'shuffling', 'swift', 'flowing', 'bouncy', 'indigenous', 'fitting', 'foolproof', 'foreign', 'ethnic', 'homegrown', 'regional', 'cosmopolitan', 'multicultural', 'traditional', 'national', 'international', 'alien', 'native', 'immigrant', 'domestic', 'imported', 'aborigine', 'heritage', 'exotic', 'factual', 'priceless', 'local', 'impoverished', 'disadvantaged', 'accomplished', 'precious', 'satisfactory', 'respected', 'safe', 'significant', 'destitute', 'successful', 'anonymous', 'influential', 'acrobatic', 'privileged', 'cherished', 'rarefied', 'underprivileged', 'reversible', 'honored', 'scalable', 'wealthy', 'prosperous', 'disrespected', 'deprived', 'affluent', 'sustainable', 'poor', 'renewable', 'unaccomplished', 'lush', 'unsuccessful', 'opulent', 'elite', 'powerless', 'needy', 'reputable', 'tasteful', 'functional', 'utilitarian', 'productive', 'uncomplicated', 'purposeful', 'intentional', 'valuable', 'invaluable', 'vivid', 'irreplaceable', 'healthy', 'esteemed', 'compliant', 'exclusive', 'costly', 'uniform', 'coveted', 'homogeneous', 'ambiguous', 'luxurious', 'worthwhile', 'expensive', 'compelling', 'constructive', 'horrendous', 'beneficial', 'profitable', 'tactical', 'abandoned', 'balanced', 'beneficent', 'usable', 'advantageous', 'effective', 'serviceable', 'picturesque', 'reusable', 'strategic', 'abhorrent', 'efficient', 'useful', 'precise', 'magical', 'striped', 'responsive', 'handy', 'abnormal', 'plaid', 'groggy', 'notable', 'spotted', 'variegated', 'suitable', 'timely', 'personalized', 'checkered', 'hurtful', 'abstract', 'instrumental', 'meaningful', 'extensible', 'scandalous', 'hygienic', 'tartan', 'dotted', 'marbled', 'secretive', 'paisley', 'atrocious', 'scientific', 'scholarly', 'idyllic', 'seamless', 'illogical'
]

def filter_tags(tags: str, input_set: Set[str]) -> str:
    tags_list = tags.split(', ')
    tags_set = set(tags_list)
    filtered_tags = tags_set - input_set
    return ', '.join(filtered_tags)


class CustomInterrogate(Interrogator):
    def __init__(self, config):
        super().__init__(config)
        self.table = LabelTable(clip_adj, 'terms', self)




    def custom_interrogate_fast(self, image: Image, max_flavors: int=8, caption: Optional[str]=None) -> str:
        """Fast mode simply adds the top ranked terms after a caption. It generally results in 
        better similarity between generated prompt and image than classic mode, but the prompts
        are less readable."""
        image_features = self.image_to_features(image)
        merged = _merge_tables([self.table], self)
        tops = merged.rank(image_features, max_flavors)
        return _truncate_to_fit(caption + ", " + ", ".join(tops), self.tokenize)

    def custom_interrogate(self, image: Image, min_flavors: int=8, max_flavors: int=8, caption: Optional[str]=None) -> str:
        image_features = self.image_to_features(image)

        merged = _merge_tables([self.table], self)
        flaves = merged.rank(image_features, self.config.flavor_intermediate_count)
        best_prompt, best_sim = caption, self.similarity(image_features, caption)
        best_prompt = self.chain(image_features, flaves, best_prompt, best_sim, min_count=0, max_count=5, desc="Flavor chain")

        fast_prompt = self.custom_interrogate_fast(image, max_flavors, caption)
        candidates = [caption, fast_prompt, best_prompt]
        return candidates[np.argmax(self.similarities(image_features, candidates))]



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ci = CustomInterrogate(Config(device=device,caption_model_name=None,clip_model_name='ViT-L-14/datacomp_xl_s13b_b90k'))


def _merge_tables(tables: List[LabelTable], ci: Interrogator) -> LabelTable:
    m = LabelTable([], None, ci)
    for table in tables:
        m.labels.extend(table.labels)
        m.embeds.extend(table.embeds)
    return m

def _prompt_at_max_len(text: str, tokenize) -> bool:
    tokens = tokenize([text])
    return tokens[0][-1] != 0

def _truncate_to_fit(text: str, tokenize) -> str:
    parts = text.split(', ')
    new_text = parts[0]
    for part in parts[1:]:
        if _prompt_at_max_len(new_text + part, tokenize):
            break
        new_text += ', ' + part
    return new_text

def process_clustering(image_info_list: List[Dict[str, Optional[str]]], tags_list, n_clusters, cluster_prefix, args):

    def extract_text_features(tags_list: List[str]) -> Tuple[np.ndarray, List[str]]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(', '), token_pattern=None)
        X = vectorizer.fit_transform(tags_list).toarray()
        feature_names = vectorizer.get_feature_names_out().tolist()
        return X, feature_names

    def perform_clustering(X: np.ndarray, n_clusters: int, model_name: str) -> np.ndarray:
        from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, OPTICS
        if model_name == "K-Means聚類":
            model = KMeans(n_clusters=n_clusters, n_init=8)
        elif model_name == "Spectral譜聚類":
            model = SpectralClustering(n_clusters=n_clusters, affinity='cosine')
        elif model_name == "Agglomerative層次聚類":
            model = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
        elif model_name == "OPTICS聚類":
            model = OPTICS(metric="cosine", min_samples=n_clusters)
        else:
            raise ValueError(f"不支持的聚類模型: {model_name}")
        return model.fit_predict(X)

    def cluster_feature_analysis(X: np.ndarray, y_pred: np.ndarray, feature_names: List[str], clusters_ID: np.ndarray) -> List[dict]:
        from sklearn.feature_selection import SelectKBest, chi2
        import pandas as pd
        cluster_feature_tags_list = []
        for i in tqdm(clusters_ID, desc="分析聚類特徵"):
            temp_pred = y_pred.copy()
            temp_pred[temp_pred != i] = i + 1
            k = min(10, X.shape[1])
            selector = SelectKBest(chi2, k=k)
            X_shifted = X - X.min(axis=0)
            X_selected = selector.fit_transform(X_shifted, temp_pred)
            X_selected_index = selector.get_support(indices=True)
            tags_selected = np.array(feature_names)[X_selected_index]
            cluster_df = pd.DataFrame(X_shifted[temp_pred == i], columns=feature_names)
            cluster_tags_df = cluster_df.iloc[:, X_selected_index]
            mean_values = cluster_tags_df.mean(axis=0)
            prompt_tags_list = mean_values.nlargest(10).index.tolist()
            cluster_feature_tags_list.append({"prompt": prompt_tags_list})
        return cluster_feature_tags_list

    def is_nsfw(tags: str) -> bool:
        """檢查標籤是否符合 NSFW 黑名單"""
        nsfw_blacklist = ['.*nude.*$', '.*penis.*$', '.*nipple.*$', '.*anus.*$', '.*sex.*$']
        for pattern in nsfw_blacklist:
            if re.search(pattern, tags, re.IGNORECASE):
                return True
        return False
        
    def is_clustering(tags: str, input_set: Set[str]) -> bool:
        tags_list = tags.split(', ')
        tags_set = set(tags_list)
        intersection = tags_set.intersection(input_set)
        if len(intersection) >= 2:
            return True
        return False

    def update_clusters(image_info_list: List[Dict[str, Optional[str]]], y_pred: np.ndarray, cluster_feature_tags_list: List[dict], cluster_prefix: str):
        cluster_counts = np.bincount(y_pred)
        non_empty_clusters = [i for i in range(len(cluster_counts))]
        sorted_clusters = sorted(non_empty_clusters, key=lambda x: cluster_counts[x], reverse=True)

        for idx, cluster_id in enumerate(sorted_clusters):
            nsfw = False
            cluster_prompt = ', '.join(cluster_feature_tags_list[cluster_id]['prompt'])
            cluster_name = 'no'
            if idx < 26:
                cluster_name = f"{cluster_prefix}{chr(97 + idx)}"

            for image_info in image_info_list:
                if y_pred[image_info_list.index(image_info)] == cluster_id:
                    if cluster_name != 'no':
                        image_info[f'{cluster_prefix}cluster_name'] = cluster_name
                    image_info[f'{cluster_prefix}cluster_prompt'] = cluster_prompt
            
    X, feature_names = extract_text_features(tags_list)
    if len(tags_list) > 0:
        y_pred = perform_clustering(X, n_clusters, args.cluster_model_name)
        clusters_ID = np.unique(y_pred)
        cluster_feature_tags_list = cluster_feature_analysis(X, y_pred, feature_names, clusters_ID)

        update_clusters(image_info_list, y_pred, cluster_feature_tags_list, cluster_prefix)

def process_image(image_path, args):
    def resize_image(image_path, max_size=512):
        """
        縮小圖像使其最大邊不超過 max_size，返回縮小後的圖像數據
        """
        image = Image.open(image_path)
        if max(image.width, image.height) > max_size:
            if image.width > image.height:
                new_width = max_size
                new_height = int(max_size * image.height / image.width)
            else:
                new_height = max_size
                new_width = int(max_size * image.width / image.height)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        return image
        
    def generate_special_text(image_path, folder_name, args, features=None, chars=None):
        """
        根據 features, image_path 和 parent_folder 生成 special_text。
        """
        def has_reverse_name(name_set, name):
            name_parts = name.split()
            if len(name_parts) == 2:
                reverse_name = f"{name_parts[1]} {name_parts[0]}"
                if reverse_name in name_set:
                    return True
            return False
        
        base_file_name = os.path.splitext(image_path)[0]
        boorutag_path = None
        boorutag = ""
        styletag = None
        chartag_from_folder = ""
        concept_tag = ""
        # 查找 boorutag 文件路徑
        for ext in ['.jpg.boorutag', '.png.boorutag']:
            potential_path = base_file_name + ext
            if os.path.exists(potential_path):
                boorutag_path = potential_path
                break

        chartags = set()

        # 獲取 parent_folder 並添加 chartag_from_folder
        parent_folder = Path(image_path).parent.name
        if folder_name and "_" in parent_folder and parent_folder.split("_")[0].isdigit():
            if not args.not_char:
                chartag_from_folder = parent_folder.split('_')[1].replace('_', ' ').strip()
                chartags.add(chartag_from_folder)
            else:
                concept_tag = f"{parent_folder.split('_')[1].replace('_', ' ').strip()} is main concept of the whole image"
                
        # 處理 boorutag 文件內容
        if boorutag_path:
            try:
                with open(boorutag_path, 'r', encoding='cp950') as file:
                    lines = file.readlines()
                    first_line = lines[0]
                    first_line_cleaned = re.sub(r'\(.*?\)', '', first_line)
                    for tag in first_line_cleaned.split(','):
                        cleaned_tag = tag.replace('\\', '').replace('_', ' ').strip()
                        if not has_reverse_name(chartags, cleaned_tag):
                            chartags.add(cleaned_tag)
                    if len(lines) >= 19:
                        artisttag = lines[6].strip()
                        boorutag = lines[18].strip()
                        boorutag_tags = drop_overlap_tags(boorutag.split(', '))
                        boorutag_tags_cleaned = [tag for tag in boorutag_tags if tag.replace(' ', '_') not in features.keys()]
                        boorutag = ', ' + ', '.join(boorutag_tags_cleaned)                
            except Exception as e:
                # 讀取文件或處理過程中發生錯誤
                pass

        # 處理 chars.keys()
        if chars:
            for key in chars.keys():
                cleaned_key = re.sub(r'\(.*?\)', '', key).replace('\\', '').replace('_', ' ').strip()
                if not has_reverse_name(chartags, cleaned_key):
                    chartags.add(cleaned_key)

        # 將 chartags 轉換為列表並隨機打亂
        chartags = list(chartags)
        random.shuffle(chartags)
        if chartag_from_folder and features and "solo" in features:
            return f"a character {chartag_from_folder} in this image", boorutag
        
        if not chartag_from_folder and features and chartags and "solo" in features:
            return f"{concept_tag} a character {' '.join(chartags)} in this image" if chartags else "", boorutag
        
        if chartags:
            if len(chartags) == 1:
                chartags.append('anonamos')    
            return f'{concept_tag}the characters in this image are {" and ".join(chartags)}', boorutag
        
        return f'{concept_tag}{chartag_from_folder}', boorutag
            
            
    def process_features(features: dict) -> (dict, str):
        """
        處理features字典，移除指定模式的鍵值對並生成keep_tags字串。
        
        參數:
        features (dict): 包含特徵的字典。

        返回:
        (dict, str): 返回處理後的features字典和keep_tags字串。
        """

        keep_tags_set = set()

        keys = list(features.keys())
        keys_to_delete = []

        for pattern in patterns_to_keep:
            regex = re.compile(pattern)
            for key in keys:
                if regex.match(key):
                    keep_tags_set.add(key.replace('_', ' '))
                    keys_to_delete.append(key)
        
        for key in keys_to_delete:
            if key in features:
                del features[key]
        
        keep_tags = ', '.join(sorted(keep_tags_set)).rstrip(', ')
        
        return features, keep_tags
        
        
    folder_name = args.folder_name
    tag_file_path = Path(image_path).with_suffix('').with_suffix('.txt')
   
    try:
        image_resize = resize_image(image_path)

        rating, features, chars = get_wd14_tags(image_resize, character_threshold=0.7, general_threshold=0.2682, model_name="ConvNext_v3",drop_overlap=True)
        features, keep_tags = process_features(features)
        rating = max(rating, key=rating.get)
        tags_text = f"|||{tags_to_text(features, use_escape=True, use_spaces=True)}\n\n\n\n"

        special_text, boorutag = generate_special_text(image_path, folder_name, args, features, chars)
        if rating:
            special_text += f", rating:{rating}"
        if keep_tags:
            special_text += f", {keep_tags}"
        tags_lines = tags_text.split('\n')
        tags_text = '\n'.join(f"{special_text}, {line}" for line in tags_lines)
        #tags_text = tags_text.replace("|||,",f"{boorutag}|||,")

        with open(tag_file_path, 'w', encoding='utf-8') as f:
            f.write(tags_text.lower())        
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")



def process_subfolder(subfolder_path: str, args, md_filepath: str):

    def read_images_and_tags(images_dir: str, file_ext: str = '.txt') -> List[Dict[str, Optional[str]]]:

        def whitelist_tags(tags: str, input_set: Set[str]) -> str:
            tags_list = tags.split(', ')
            tags_set = set(tags_list)
            intersection = tags_set.intersection(input_set)
            return ', '.join(intersection)

        def repeat_tags(tags: str, input_set: Set[str], repeat_count: int = 2) -> str:
            tags_list = tags.split(', ')
            tags_set = set(tags_list)
            intersection = tags_set.intersection(input_set)
            result_tags = tags_list
            for i in range(repeat_count):
                result_tags.extend(intersection)
            return ', '.join(result_tags)

        image_info_list = []
        image_base_names = set(os.path.splitext(file)[0] for ext in IMAGE_EXTENSIONS for file in glob.glob(os.path.join(images_dir, f"*{ext.lower()}")) + glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
        
        for base_name in image_base_names:
            txt_file = os.path.join(images_dir, f"{base_name}{file_ext}")
            image_path = None
            for ext in IMAGE_EXTENSIONS:
                possible_image_path = os.path.join(images_dir, f"{base_name}{ext}")
                if os.path.exists(possible_image_path):
                    image_path = possible_image_path
                    break
            
            if image_path and os.path.exists(txt_file):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if '|||' in first_line:
                        tags = first_line.split('|||')[1].strip() if len(first_line.split('|||')) > 1 else ''
                    else:
                        tags = first_line.split(', ', 1)[1] if ', ' in first_line else first_line
                    image_info_list.append({
                        'path': image_path,
                        'costume': repeat_tags(tags, not_scene_tags),
                        'appearance': repeat_tags(tags, appearance_tags),
                        'scene': tags,
                        'all_tags': tags,
                        'costume_cluster_name': None,
                        'costume_cluster_prompt': None,
                        'appearance_cluster_name': None,
                        'appearance_cluster_prompt': None,
                        'scene_cluster_name': None,
                        'scene_cluster_prompt': None                 
                    })
        
        return image_info_list

    def insert_cluster_text_to_txt(info: Dict[str, Optional[str]], args):
        def contains_color(tag: str) -> bool:
            colors = {'red', 'orange', 'yellow', 'green', 'blue', 'aqua', 'purple', 'brown', 'pink', 'black', 'white', 'grey', 'dark-', 'light ', 'pale', 'blonde'}
            return any(color in tag for color in colors)

        txt_filepath = os.path.splitext(info['path'])[0] + '.txt'
        info_cluster_name = info.get(f'{args.dir_mode}_cluster_name', '')

        # 檢查每個標籤是否為 None，並組合有效的標籤
        costume_cluster_prompt = info.get('costume_cluster_prompt', '')
        appearance_cluster_prompt = info.get('scene_appearance_prompt', '')
        scene_cluster_prompt = info.get('scene_cluster_prompt', '')

        # 分割標籤，過濾空標籤
        cluster_costume_tags = ', '.join(filter(None, [costume_cluster_prompt])).split(', ')
        cluster_appearance_tags = ', '.join(filter(None, [appearance_cluster_prompt])).split(', ')
        cluster_scene_tags = ', '.join(filter(None, [scene_cluster_prompt])).split(', ')

        with open(txt_filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        if not lines:
            return  

        new_tags = info['all_tags'].split(', ')
        cluster_text = ''
        new_caption = ''
        if info_cluster_name is not None:
            cluster_text += f'{info_cluster_name}, '

        for tag in new_tags:
            if not contains_color(tag) and f"{tag}" not in cluster_text and f"{tag}" not in new_caption:
                if tag in cluster_costume_tags:
                    cluster_text += f'{tag}, '
#                elif tag in cluster_appearance_tags:
#                    cluster_text += f'{tag}, '
                elif tag in keep_tags:
                    cluster_text += f'{tag}, '
#                elif tag in cluster_scene_tags:
#                    cluster_text += f'{tag}, '# if 'explicit' in lines[0] else f'{tag}, ' if 'questionable' in lines[0] else ''
                else:
                    new_caption += f'{tag}, '
                    
#        cluster_text += f"{ci.custom_interrogate_fast(image=Image.open(info['path']), caption=cluster_text)}, "
#
        for i in range(len(lines)):
            if i < 3:
                line = lines[i].strip()
                if new_caption and i == 0:
                    line = line.replace(info['all_tags'], new_caption)
                parts = line.split(', ', 1)
                if len(parts) > 1:
                    lines[i] = f"{parts[0]}, {cluster_text}{parts[1]}"
                else:
                    lines[i] = f"{line} {cluster_text}"   
        
        lines = [line.strip() for line in lines if line.strip()]        

        with open(txt_filepath, 'w', encoding='utf-8') as file:
            file.write('\n'.join(lines))
        
    def copy_or_move_clusters(subfolder_path: str, subfolder_name: str, image_info_list: List[Dict[str, Optional[str]]], repeats: int, name_from_folder: str, args):
        cluster_images = {}
        for info in image_info_list:
            cluster_name = info[f'{args.dir_mode}_cluster_name'] 
            if cluster_name:
                if cluster_name not in cluster_images:
                    cluster_images[cluster_name] = []
                cluster_images[cluster_name].append(info['path'])

        # 計算每個聚類要複製幾份
        max_cluster_size = max(len(images) for images in cluster_images.values())
        num_subfolder_images = len(image_info_list) * repeats
        extra_repeats = max(1, int(math.ceil(num_subfolder_images / (max_cluster_size * len(cluster_images)))))

        for cluster_name, images in tqdm(cluster_images.items(), desc="移動或複製檔案"):
            num_copies = int(max_cluster_size / len(images))
            if num_copies > 15:
                continue
            if args.copy_cluster:
                extra_folder_name = f"{extra_repeats}_{name_from_folder} extra hard link"
                extra_folder_path = os.path.join(os.path.dirname(subfolder_path), extra_folder_name)
                os.makedirs(extra_folder_path, exist_ok=True)

                for i in range(num_copies):
                    for img_path in images:
                        img_base_name = os.path.basename(img_path)
                        new_img_path = os.path.join(extra_folder_path, f"{i}_{img_base_name}")
                        try:
                            os.link(img_path, new_img_path)
                        except (FileExistsError, OSError) as e:
                            if isinstance(e, OSError):
                                print(f"硬連結失敗 {img_path} -> {new_img_path}: {e}, 將使用複製")
                                shutil.copy(img_path, new_img_path)
                                extra_folder_name = f"{extra_repeats}_{name_from_folder} extra copy"
                                extra_folder_path = os.path.join(os.path.dirname(subfolder_path), extra_folder_name)
                                os.makedirs(extra_folder_path, exist_ok=True)
                            else:
                                print(f"文件已存在: {new_img_path}")

                        txt_file = os.path.splitext(img_path)[0] + '.txt'
                        if os.path.exists(txt_file):
                            new_txt_path = os.path.join(extra_folder_path, f"{i}_{os.path.basename(txt_file)}")
                            try:
                                os.link(txt_file, new_txt_path)
                            except (FileExistsError, OSError) as e:
                                if isinstance(e, OSError):
                                    print(f"硬連結失敗 {txt_file} -> {new_txt_path}: {e}, 將使用複製")
                                    shutil.copy(txt_file, new_txt_path)
                                    extra_folder_name = f"{extra_repeats}_{name_from_folder} extra copy"
                                    extra_folder_path = os.path.join(os.path.dirname(subfolder_path), extra_folder_name)
                                    os.makedirs(extra_folder_path, exist_ok=True)
                                else:
                                    print(f"文件已存在: {new_txt_path}")

                        npz_file = os.path.splitext(img_path)[0] + '.npz'
                        if os.path.exists(npz_file):
                            new_npz_path = os.path.join(extra_folder_path, f"{i}_{os.path.basename(npz_file)}")
                            try:
                                os.link(npz_file, new_npz_path)
                            except (FileExistsError, OSError) as e:
                                if isinstance(e, OSError):
                                    print(f"硬連結失敗 {npz_file} -> {new_npz_path}: {e}, 將使用複製")
                                    shutil.copy(npz_file, new_npz_path)
                                    extra_folder_name = f"{extra_repeats}_{name_from_folder} extra copy"
                                    extra_folder_path = os.path.join(os.path.dirname(subfolder_path), extra_folder_name)
                                    os.makedirs(extra_folder_path, exist_ok=True)
                                else:
                                    print(f"文件已存在: {new_npz_path}")

            if args.move_cluster:
                cluster_dir = os.path.join(subfolder_path, f"{num_copies}_{cluster_name}")
                os.makedirs(cluster_dir, exist_ok=True)
                for img_path in images:
                    try:
                        shutil.move(img_path, os.path.join(cluster_dir, os.path.basename(img_path)))
                        txt_file = os.path.splitext(img_path)[0] + '.txt'
                        if os.path.exists(txt_file):
                            shutil.move(txt_file, os.path.join(cluster_dir, os.path.basename(txt_file)))
                        npz_file = os.path.splitext(img_path)[0] + '.npz'
                        if os.path.exists(npz_file):
                            shutil.move(npz_file, os.path.join(cluster_dir, os.path.basename(npz_file)))
                    except FileNotFoundError as e:
                        print(f"文件未找到: {e.filename}")

        if not args.copy_cluster and args.move_cluster:
            root_dir = os.path.join(subfolder_path, '1_')
            os.makedirs(root_dir, exist_ok=True)
            
            for file_path in glob.glob(os.path.join(subfolder_path, '*')):
                if os.path.isfile(file_path):
                    try:
                        shutil.move(file_path, os.path.join(root_dir, os.path.basename(file_path)))
                    except FileNotFoundError as e:
                        print(f"文件未找到: {e.filename}")

    def write_cluster_results_to_md(md_filepath: str, subfolder_path: str, image_info_list: List[Dict[str, Optional[str]]]):
        with open(md_filepath, 'a', encoding='utf-8') as md_file:
            md_file.write(f"# 聚類結果 - {subfolder_path}\n")
            md_file.write(f"總圖片數: {len(image_info_list)}\n")
            clusters = {}
            clusters_prompts = []
            cluster_prefixes = ['costume_', 'appearance_', 'scene_']
            for cluster_prefix in cluster_prefixes:
                for info in image_info_list:
                    if info[f'{cluster_prefix}cluster_name'] is not None:
                        if info[f'{cluster_prefix}cluster_name'] not in clusters:
                            clusters[info[f'{cluster_prefix}cluster_name']] = {
                                'count': 0,
                                'prompt': info[f'{cluster_prefix}cluster_prompt']
                            }
                        clusters[info[f'{cluster_prefix}cluster_name']]['count'] += 1
                        

            # 排序非 None 的聚類名
            sorted_cluster_names = natsorted(clusters.keys())
            print("\n")
            for cluster_name in sorted_cluster_names:
                cluster_data = clusters[cluster_name]
                print(f"最終聚類名稱: {cluster_name}")
                print(f"聚類標籤: {cluster_name}, {cluster_data['prompt']}")
                print(f"聚類張數: {cluster_data['count']}")
                print("")    
                md_file.write(f"## {cluster_name}\n")
                md_file.write(f"{cluster_name}, {cluster_data['prompt']}\n")
                md_file.write(f"聚類張數: {cluster_data['count']}\n")
                md_file.write("\n")
                clusters_prompts.append(cluster_data['prompt'])
                
    subfolder_name = os.path.basename(subfolder_path)
    if "_" not in subfolder_name or not subfolder_name.split("_")[0].isdigit() or ' extra ' in subfolder_name:
        print(f"跳過不符合命名規則的子文件夾: {subfolder_path}")
        return

    repeats = int(subfolder_name.split("_")[0])
    name_from_folder = subfolder_name.split('_')[1].replace('_', ' ').strip()

    print(f"處理子文件夾: {subfolder_name}")
    
#    image_base_names = set(os.path.splitext(file)[0] for ext in IMAGE_EXTENSIONS for file in glob.glob(os.path.join(subfolder_path, f"*{ext.lower()}")) + glob.glob(os.path.join(subfolder_path, f"*{ext.upper()}")))
#    image_paths = []    
#    for base_name in image_base_names:
#        txt_file = os.path.join(subfolder_path, f"{base_name}.txt")
#        
#        for ext in IMAGE_EXTENSIONS:
#            possible_image_path = os.path.join(subfolder_path, f"{base_name}{ext}")
#            if os.path.exists(possible_image_path):
#                image_paths.append(possible_image_path)
#                break
#
#    for image_path in tqdm(image_paths, desc="處理圖片"):
#        try:
#            process_image(image_path, args)
#        except Exception as e:
#            print(f"Failed to process image {image_path}: {e}")
    
    image_info_list = read_images_and_tags(subfolder_path)
    if not image_info_list or len(image_info_list) < 3:
        print(f"子文件夾 {subfolder_path} 沒有有效的標籤，跳過該文件夾。")
        return

    print("開始聚類...")
    
    costume_info_list = [info for info in image_info_list if 'solo' in info['all_tags'] and 'completely nude' not in info['all_tags']]
    costume_tags_list = [info['costume'] for info in costume_info_list]
    n_clusters = min(300, math.ceil(len(costume_info_list) / 5) + 1)
    if len(costume_tags_list) > 0:
        process_clustering(costume_info_list, costume_tags_list, n_clusters, 'costume_', args)
        print("服裝聚類完成")

    appearance_info_list = [info for info in image_info_list if 'doors' in info['all_tags'] and 'solo' in info['all_tags'] and info['costume_cluster_name'] is None]
    appearance_tags_list = [info['appearance'] for info in appearance_info_list]
    n_clusters = min(300, math.ceil(len(appearance_info_list) / 5) + 1)
    if len(appearance_tags_list) > 0:
        process_clustering(appearance_info_list, appearance_tags_list, n_clusters, 'appearance_', args)
        print("外型聚類完成")
        
    scene_info_list = [info for info in image_info_list if info['costume_cluster_name'] is None and info['appearance_cluster_name'] is None]
    scene_tags_list = [info['scene'] for info in scene_info_list]
    n_clusters = min(300, math.ceil(len(scene_info_list) / 5) + 1)     
    if len(scene_tags_list) > 0:        
        process_clustering(scene_info_list, scene_tags_list, n_clusters, 'scene_', args)
        print("場景聚類完成")
        
    if not args.dry_run:
        for info in tqdm(image_info_list, desc="修改文本"):
            insert_cluster_text_to_txt(info, args)

    copy_or_move_clusters(subfolder_path, subfolder_name, image_info_list, repeats, name_from_folder, args)

    write_cluster_results_to_md(md_filepath, subfolder_path, image_info_list)

def main():
    parser = argparse.ArgumentParser(description="聚類分析腳本")
    parser.add_argument("--folder_name", action="store_true", help="啟用特殊資料夾名稱處理")
    parser.add_argument("--not_char", action="store_true", help="非角色")
    parser.add_argument('--cluster_model_name', choices=['K-Means聚類', 'Spectral譜聚類', 'Agglomerative層次聚類', 'OPTICS聚類'], default='Agglomerative層次聚類', help='聚類模型名稱')
    parser.add_argument('--dry_run', action='store_true', help='不輸出文本')
    parser.add_argument('--move_cluster', action='store_true', help='移動到子資料夾的聚類文件夾')
    parser.add_argument('--copy_cluster', action='store_true', help='複製到子資料夾的extra文件夾')
    parser.add_argument('--dir_mode', choices=['costume', 'appearance', 'scene'], default='costume', help='檔案模式：依照服裝、外表或場景聚類服裝')
    args = parser.parse_args()
    
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    subfolders = [f.path for f in os.scandir(parent_dir) if f.is_dir()]

    md_filepath = os.path.join(parent_dir, "cluster_results.md")
    with open(md_filepath, 'w', encoding='utf-8') as md_file:
        md_file.write("# 聚類結果\n\n")      

    for subfolder in subfolders:
        #try:
        process_subfolder(subfolder, args, md_filepath)
        #except:
        #    print(f"{subfolders}處理出錯 略過")


if __name__ == "__main__":
    main()