import subprocess
import os
import sys
import shutil
import platform
import requests
import argparse
from io import BytesIO
from pathlib import Path
from glob import glob
from tqdm import tqdm
from PIL import Image
from datetime import datetime, timedelta
from transformers import AutoProcessor, AutoModelForCausalLM, CLIPProcessor, CLIPModel
import requests
import copy
import inflect
import re
import random
import torch
import torch.nn.functional as F
from model import longclip
import ftfy
import onnxruntime
import fnmatch
from imgutils.tagging import get_wd14_tags, tags_to_text, drop_blacklisted_tags, drop_basic_character_tags, drop_overlap_tags
from imgutils.validate import anime_dbrating

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().to(device).half()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
p = inflect.engine()
clip_model, clip_preprocess = longclip.load("./checkpoints/Long-ViT-L-14-GmP-ft-state_dict.pt", device=device)

chartags = {
    'long hair', 'short hair', 'blue eyes', 'large breasts', 'blonde hair', 'brown hair', 'black hair', 'hair ornament', 'red eyes', 'hat', 'bow', 'animal ears', 'ribbon', 'hair between eyes', 'very long hair', 'twintails', 'medium breasts', 'brown eyes', 'green eyes', 'blue hair', 'purple eyes', 'tail', 'yellow eyes', 'white hair', 'pink hair', 'grey hair', 'ahoge', 'braid', 'hair ribbon', 'purple hair', 'ponytail', 'multicolored hair', 'sidelocks', 'hair bow', 'earrings', 'red hair', 'small breasts', 'hairband', 'horns', 'wings', 'green hair', 'glasses', 'pointy ears', 'hairclip', 'medium hair', 'fang', 'dark skin', 'cat ears', 'blunt bangs', 'hair flower', 'pink eyes', 'hair bun', 'mole', 'hair over one eye', 'rabbit ears', 'orange hair', 'black eyes', 'two-tone hair', 'streaked hair', 'huge breasts', 'halo', 'red bow', 'twin braids', 'side ponytail', 'animal ear fluff', 'red ribbon', 'aqua eyes', 'dark-skinned female', 'parted bangs', 'two side up', 'v-shaped eyebrows', 'grey eyes', 'orange eyes', 'cat tail', 'symbol-shaped pupils', 'eyelashes', 'lips', 'black headwear', 'mole under eye', 'fox ears', 'maid headdress', 'shiny skin', 'fake animal ears', 'black bow', 'single braid', 'neck ribbon', 'black ribbon', 'gradient hair', 'double bun', 'floating hair', 'aqua hair', 'colored skin', 'swept bangs', 'facial hair', 'heterochromia', 'white headwear', 'blue bow', 'fox tail', 'witch hat', 'low twintails', 'one side up', 'headband', 'horse ears', 'beret', 'wavy hair', 'fangs', 'headphones', 'hair intakes', 'facial mark', 'thick eyebrows', 'horse girl', 'headgear', 'muscular male', 'heart-shaped pupils', 'bob cut', 'drill hair', 'sunglasses', 'dark-skinned male', 'light brown hair', 'wolf ears', 'black hairband', 'eyepatch', 'scrunchie', 'white bow', 'demon girl', 'cat girl', 'mob cap', 'magical girl', 'eyes visible through hair', 'demon horns', 'single hair bun', 'high ponytail', 'x hair ornament', 'fox girl', 'blue ribbon', 'grabbing another\'s breast', 'antenna hair', 'hat ribbon', 'crown', 'pink bow', 'spiked hair', 'bat wings', 'ear piercing', 'slit pupils', 'bright pupils', 'monster girl', 'rabbit tail', 'tassel', 'head wings', 'short twintails', 'messy hair', 'horse tail', 'straight hair', 'feathered wings', 'hat bow', 'multiple tails', 'extra ears', 'eyewear on head', 'demon tail', 'dog ears', 'pale skin', 'red headwear', 'white ribbon', 'between breasts', 'colored inner hair', 'hair over shoulder', 'skin fang', 'mole under mouth', 'side braid', 'third eye', 'scar on face', 'baseball cap', 'beard', 'blue headwear', 'peaked cap', 'glowing eyes', 'white pupils', 'semi-rimless eyewear', 'low ponytail', 'twin drills', 'yellow bow', 'wolf tail', 'eyeshadow', 'french braid', 'no headwear', 'tokin hat', 'crossed bangs', 'black wings', 'green bow', 'single horn', 'dragon horns', 'drinking glass', 'hair scrunchie', 'santa hat', 'pink ribbon', 'half updo', 'freckles', 'demon wings', 'topless male', 'single earring', 'low-tied long hair', 'white skin', 'hair rings', 'mature male', 'unworn headwear', 'mole on breast', 'black-framed eyewear', 'short ponytail', 'purple bow', 'round eyewear', 'angel wings', 'goggles on head', 'braided ponytail', 'red-framed eyewear', 'curly hair', 'raised eyebrows', 'hat ornament', 'dragon girl', 'faceless male', 'asymmetrical hair', 'dog tail', 'yellow ribbon', 'top hat', 'sun hat', 'furry female', 'white hairband', 'asymmetrical bangs', 'fake tail', 'blood on face', 'star hair ornament', 'under-rim eyewear', 'white wings', 'mature female', 'multicolored eyes', 'colored eyelashes', 'rabbit girl', 'hoop earrings', 'bouncing breasts', 'unworn hat', 'tentacle hair', 'eyebrows hidden by hair', 'green headwear', 'wolf girl', 'light blue hair', 'mini hat', 'military hat', 'brown headwear', 'dragon tail', 'striped bow', 'tress ribbon', 'pink lips', 'short eyebrows', 'scar across eye', 'mustache', 'folded ponytail', 'dog girl', 'furry male', 'blue skin', 'heart hair ornament', 'muscular female', 'red hairband', 'hime cut', 'mouse ears', 'bandaid on face', 'nurse cap', 'purple ribbon', 'butterfly hair ornament', 'straw hat', 'green ribbon', 'visor cap', 'orange bow', 'stud earrings', 'licking lips', 'bags under eyes', 'low wings', 'long bangs', 'eyeliner', 'red lips', 'fake horns', 'back bow', 'crown braid', 'tail ornament', 'hanging breasts', 'sailor hat', 'hair behind ear', 'cabbie hat', 'flipped hair', 'single side bun', 'absurdly long hair', 'frog hair ornament', 'on head', 'fairy wings', 'star-shaped pupils', 'bird wings', 'hair over eyes', 'cow ears', 'glass', 'food-themed hair ornament', 'pink headwear', 'wrist scrunchie', 'black horns', 'headdress', 'feather hair ornament', 'tinted eyewear', 'ringed eyes', 'mask on head', 'covered eyes', 'horn ornament', 'cow horns', 'mini crown', 'very short hair', 'blue hairband', 'green skin', 'blue halo', 'tiger ears', 'symbol in eye', 'wet hair', 'purple headwear', 'flat cap', 'wine glass', 'snake hair ornament', 'cone hair bun', 'curled horns', 'ice wings', 'bald', 'mechanical halo', 'red horns', 'animal hat', 'raccoon ears', 'pink halo', 'unworn eyewear', 'lolita hairband', 'star earrings', 'crescent hair ornament', 'mouse tail', 'leg ribbon', 'garrison cap', 'white eyes', 'deep skin', 'frilled bow', 'tilted headwear', 'animal on head', 'grey skin', 'ear ornament', 'asymmetrical wings', 'two tails', 'facial tattoo', 'crescent hat ornament', 'rolling eyes', 'toned male', 'no pupils', 'glowing eye', 'fish tail', 'constricted pupils', 'split-color hair', 'leaf hair ornament', 'rabbit hair ornament', 'red skin', 'chest hair', 'leaf on head', 'goat horns', 'necktie between breasts', 'raccoon tail', 'multicolored skin', 'polka dot bow', 'ears through headwear', 'purple skin', 'heart earrings', 'double-parted bangs', 'dark blue hair', 'big hair', 'frilled hairband', 'hair over breasts', 'blank eyes', 'lion ears', 'sparkling eyes', 'tiger tail', 'cow girl', 'huge ahoge', 'tassel earrings', 'star hat ornament', 'braided bun', 'assertive female', 'grey headwear', 'mini top hat', 'arm ribbon', 'braided bangs', 'bear ears', 'shark tail', 'red halo', 'red eyeshadow', 'sheep horns', 'insect wings', 'rimless eyewear', 'bow hairband', 'skin-covered horns', 'yellow halo', 'anchor hair ornament', 'navel hair', 'yellow hairband', 'no eyes', 'ear bow', 'gigantic breasts', 'extra eyes', 'long braid', 'jphones', 'large bow', 'tail ribbon', 'bird ears', 'pink skin', 'cat boy', 'shark girl', 'mouse girl', 'arthropod girl', 'fur hat', 'fur-trimmed headwear', 'raised eyebrow', 'black skin', 'frilled hat', 'striped ribbon', 'waist bow', 'super crown', 'low twin braids', 'crazy eyes', 'cat hair ornament', 'blue wings', 'naked ribbon', 'butterfly wings', 'multiple hair bows', 'demon boy', 'sagging breasts', 'dress bow', 'red scrunchie', 'dragon wings', 'forked eyebrows', 'armpit hair', 'footwear bow', 'purple hairband', 'multiple wings', 'wrist ribbon', 'v over eye', 'red pupils', 'pirate hat', 'towel on head', 'orange headwear', 'bow-shaped hair', 'against glass', 'leg hair', 'mini wings', 'multiple horns', 'carrot hair ornament', 'long eyelashes', 'backwards hat', 'black tail', 'red headband', 'tiger girl', 'mechanical wings', 'white horns', 'musical note hair ornament', 'unaligned breasts', 'orange ribbon', 'heart-shaped eyewear', 'small horns', 'uneven eyes', 'lion tail', 'dangle earrings', 'print bow', 'dog boy', 'raccoon girl', 'blue scrunchie', 'lion girl', 'opaque glasses', 'robot ears', 'christmas ornaments', 'biting own lip', 'framed breasts', 'wizard hat', 'cat ear headphones', 'quad tails', 'bandage over one eye', 'sheep ears', 'arms under breasts', 'diagonal bangs', 'wing hair ornament', 'perky breasts', 'bone hair ornament', 'striped tail', 'cuts', 'medical eyepatch', 'braided hair rings', 'multicolored wings', 'rectangular eyewear', 'purple wings', 'squirrel ears', 'ear ribbon', 'black headband', 'multiple earrings', 'single hair intake', 'sheep girl', 'updo', 'bat hair ornament', 'goggles on headwear', 'horned headwear', 'white scrunchie', 'red eyeliner', 'black scrunchie', 'white headband', 'blue-framed eyewear', 'squirrel tail', 'horn bow', 'green hairband', 'horizontal pupils', 'stained glass', 'wolf boy', 'horseshoe ornament', 'chef hat', 'black lips', 'fox boy', 'multi-tied hair', 'slime girl', 'animal ear piercing', 'shark hair ornament', 'bird girl', 'gold earrings', 'tassel hair ornament', 'feather hair', 'puckered lips', 'orange hairband', 'ankle ribbon', 'flower earrings', 'grey horns', 'crescent earrings', 'yellow pupils', 'drill sidelocks', 'pink scrunchie', 'strap between breasts', 'winged hat', 'ghost tail', 'porkpie hat', 'parted hair', 'squirrel girl', 'police hat', 'over-rim eyewear', 'diagonal-striped bow', 'shower head', 'monkey tail', 'energy wings', 'wide ponytail', 'snowflake hair ornament', 'yellow scrunchie', 'brown ribbon', 'jackal ears', 'bandaged head', 'high side ponytail', 'blue lips', 'clover hair ornament', 'diamond-shaped pupils', 'long pointy ears', 'frilled ribbon', 'broken glass', 'flame-tipped tail', 'turning head', 'tiger boy', 'hair horns', 'skin fangs', 'deer ears', 'looking over eyewear', 'pink-framed eyewear', 'feather earrings', 'broken horn', 'laurel crown', 'large hat', 'flaming eye', 'pom pom hair ornament', 'grey bow', 'disembodied head', 'narrowed eyes', 'no eyewear', 'yellow skin', 'orange scrunchie', 'aqua ribbon', 'large tail', 'averting eyes', 'dreadlocks', 'character hair ornament', 'mechanical horns', 'grey-framed eyewear', 'star halo', 'cocktail glass', 'striped horns', 'multiple moles', 'curtained hair', 'cat hat', 'green lips', 'shako cap', 'buzz cut', 'dragon boy', 'alternate headwear', 'asymmetrical horns', 'short bangs', 'orange-tinted eyewear', 'cracked skin', 'yellow-framed eyewear', 'bandage on face', 'snake tail', 'thigh ribbon', 'afro', 'white-framed eyewear', 'd-pad hair ornament', 'tri tails', 'spread wings', 'school hat', 'tall female', 'bisexual female', 'cone horns', 'pink pupils', 'hair through headwear', 'mechanical tail', 'prehensile hair', 'patchwork skin', 'blue eyeshadow', 'drop earrings', 'veiny breasts', 'two-tone ribbon', 'bear hair ornament', 'bowl hat', 'gold hairband', 'spider girl', 'red-tinted eyewear', 'eyebrow cut', 'animal ear headwear', 'goat ears', 'single hair ring', 'fish hair ornament', 'dixie cup hat', 'leopard ears', 'skull earrings', 'party hat', 'blue horns', 'brushing hair', 'plaid headwear', 'white tail', 'brown hairband', 'blood from eyes', 'fiery hair', 'green halo', 'dyed bangs', 'two-tone eyes', 'wrinkled skin', 'bat ears', 'black halo', 'upturned eyes', 'bowl cut', 'bear girl', 'blue headband', 'yellow wings', 'fish girl', 'fake wings', 'x-shaped pupils', 'fake facial hair', 'flower ornament', 'pillbox hat', 'circle cut', 'yellow horns', 'body hair', 'hair ears', 'bow earrings', 'no wings', 'doughnut hair bun', 'green-framed eyewear', 'magnifying glass', 'eyewear on headwear', 'brown horns', 'plant girl', 'pink eyeshadow', 'multiple braids', 'magatama earrings', 'brown-framed eyewear', 'blue-tinted eyewear', 'cow boy', 'spiked tail', 'purple eyeshadow', 'body freckles', 'multicolored bow', 'heart tail', 'large wings', 'triangle earrings', 'rabbit boy', 'horns through headwear', 'purple-tinted eyewear', 'unusually open eyes', 'sunflower hair ornament', 'lizard tail', 'multicolored horns', 'arm between breasts', 'two-tone headwear', 'panda ears', 'fake mustache', 'expressive hair', 'purple tail', 'drawing bow', 'object through head', 'pink wings', 'blue pupils', 'transparent wings', 'purple horns', 'phoenix crown', 'artificial eye', 'grey ribbon', 'striped headwear', 'goat girl', 'tulip hat', 'crystal hair', 'aqua headwear', 'arched bangs', 'broken halo', 'mechanical ears', 'brown wings', 'leopard tail', 'grey halo', 'no eyebrows', 'notched ear', 'monkey ears', 'pink-tinted eyewear', 'fiery horns', 'uneven horns', 'jaguar ears', 'purple halo', 'sphere earrings', 'bat girl', 'candy hair ornament', 'tapir tail', 'dark halo', 'ruffling hair', 'diving mask on head', 'triangle hair ornament', 'mechanical eye', 'huge bow', 'robot girl', 'sleeve bow', 'rabbit-shaped pupils', 'dice hair ornament', 'button eyes',  'prehensile tail', 'multicolored headwear', 'green wings', 'solid eyes', 'thick lips', 'compass rose halo', 'brown tail', 'strawberry hair ornament', 'food-themed earrings', 'split ponytail', 'two-tone bow', 'neck tassel', 'lion boy', 'two-tone hairband', 'gradient skin', 'polka dot headwear', 'purple scrunchie', 'glowing wings', 'crystal earrings', 'liquid hair', 'orange skin', 'cetacean tail', 'glowing hair', 'smokestack hair ornament', 'panties on head', 'crocodilian tail', 'long tail', 'pearl earrings', 'glowing horns', 'red tail', 'print headwear', 'egg hair ornament', 'side drill', 'blue tail', 'huge eyebrows', 'hair wings', 'snake hair', 'thick eyelashes', 'swim cap', 'grey tail', 'choppy bangs', 'aviator sunglasses', 'pill earrings', 'no tail', 'pink tail', 'owl ears', 'pointy breasts', 'hat over one eye', 'full beard', 'bandaid hair ornament', 'footwear ribbon', 'grey hairband', 'coin hair ornament', 'bucket hat', 'alpaca ears', 'yellow tail', 'low-tied sidelocks', 'weasel ears', 'wrist bow', 'grey wings', 'pursed lips', 'no eyepatch', 'deer girl', 'white headdress', 'green tail', 'wing ornament', 'mismatched eyebrows', 'sleeve ribbon', 'purple-framed eyewear', 'rainbow hair', 'hedgehog ears', 'sideways hat', 'flower on head', 'coke-bottle glasses', 'fish boy', 'orange tail', 'hard hat', 'hair on horn', 'ribbon-trimmed headwear', 'multiple heads', 'flower over eye', 'yellow-tinted eyewear', 'otter ears', 'dashed eyes', 'low-braided long hair', 'arm above head', 'lace-trimmed hairband', 'four-leaf clover hair ornament', 'potara earrings', 'detached hair', 'cephalopod eyes', 'long beard', 'camouflage headwear', 'japari bun', 'star ornament', 'striped hairband', 'hat with ears', 'bunching hair', 'ears visible through hair', 'green scrunchie', 'thick mustache', 'diamond hairband', 'polka dot scrunchie', 'cherry hair ornament', 'bear tail', 'jaguar tail', 'v-shaped eyes', 'rabbit hat', 'thick beard', 'hugging tail', 'no mole', 'green-tinted eyewear', 'ornament', 'diamond hair ornament', 'wavy eyes', 'shell hair ornament', 'heart-shaped eyes', 'chain headband', 'planet hair ornament', 'pearl hair ornament', 'multicolored hairband', 'drop-shaped pupils', 'polka dot ribbon', 'ribbon braid', 'alternate wings', 'hollow eyes', 'unworn eyepatch',  'spaceship hair ornament', 'bowler hat', 'green eyeshadow', 'pumpkin hair ornament', 'spiked hairband', 'flower in eye', 'magical boy', 'behind-the-head headphones', 'plaid ribbon', 'skull ornament', 'bear boy', 'holly hair ornament', 'uneven twintails', 'folded hair', 'pig ears', 'metal skin', 'pumpkin hat', 'cut bangs', 'mole under each eye', 'clock eyes', 'reptile girl', 'hair between breasts', 'alternate hair ornament', 'licking ear', 'braiding hair', 'hexagon hair ornament', 'tri braids', 'animal ear hairband', 'solid circle pupils', 'penis to breast', 'frog girl', 'curly eyebrows', 'star-shaped eyewear', 'fiery wings', 'orange headband', 'scratching head', 'bloodshot eyes', 'green horns', 'green headband', 'single head wing', 'animal head', 'bulging eyes', 'deer tail', 'weasel girl', 'brown lips', 'lifebuoy ornament', 'frilled headwear', 'cable tail', 'safety glasses', 'leopard girl', 'wing ears', 'spade hair ornament', 'white halo', 'weasel tail', 'propeller hair ornament', 'wide oval eyes', 'otter tail', 'pom pom earrings', 'checkered bow', 'fruit hat ornament', 'starfish hair ornament', 'aqua hairband', 'crystal wings', 'object head', 'multicolored tail', 'gradient wings', 'giant male', 'purple pupils', 'torn wings', 'head on head', 'moose ears', 'pointy hat', 'hair over one breast', 'forked tail', 'lightning bolt hair ornament', 'undone neck ribbon', 'hedgehog tail', 'lop rabbit ears', 'sparse chest hair', 'pink horns', 'pokemon ears', 'ankle bow', 'bird boy', 'bandaid on head', 'implied extra ears', 'hat tassel', 'fruit on head', 'starry hair', 'sparkle hair ornament', 'long ribbon', 'rice hat', 'washing hair', 'anchor earrings', 'asymmetrical sidelocks', 'mini witch hat', 'unworn hair ornament', 'heart hair', 'arthropod boy', 'detached ahoge', 'large ears', 'aviator cap', 'monkey boy', 'female service cap', 'moth girl', 'glove bow', 'bangs', 'shiny hair', 'light purple hair', 'oni horns', 'pillow hat', 'polos crown', 'light green hair', 'monocle hair ornament', 'dark green hair', 'pouty lips', 'bunny-shaped pupils', 'bunny hatester cap', 'detached wings', 'solid oval eyes', 'cube hair ornament', 'heart ahoge', 'cross-shaped pupils', 'cross hair ornament', 'pointy hair', 'very dark skin', 'aqua bow', 'front ponytail', 'pink hairband', 'skull hair ornament', 'side braids', 'tail bow', 'cross earrings', 'horn ribbon', 'cow tail', 'floppy ears', 'two-tone skin', 'plaid bow', 'purple lips', 'single sidelock', 'solid circle eyes', 'yellow headwear', 'faceless female', 'single wing', 'brown bow', 'medium bangs', 'red wings', 'monster boy', 'mismatched pupils', 'cowboy hat', 'flower-shaped pupils', 'bird tail', 'gradient eyes', 'bursting breasts', 'animal ear head', 'hair bobbles', 'prosthetic leg', 'centaur'
}

clip_labels = ['abundant', 'accurate', 'addicted', 'adorable', 'adventurous', 'afraid', 'aggressive', 'alcoholic', 'alert', 'aloof', 'ambitious', 'ancient', 'angry', 'animated', 'annoying', 'anxious', 'arrogant', 'ashamed', 'attractive', 'auspicious', 'awesome', 'awful', 'abactinal', 'abandoned', 'abashed', 'abatable', 'abatic', 'abaxial', 'abbatial', 'abbreviated', 'abducent', 'abducting', 'aberrant', 'abeyant', 'abhorrent', 'abiding', 'abient', 'bad', 'bashful', 'beautiful', 'belligerent', 'beneficial', 'best', 'big', 'bitter', 'bizarre', 'black', 'blue', 'boring', 'brainy', 'bright', 'broad', 'broken', 'busy', 'barren', 'barricaded', 'barytic', 'basal', 'basaltic', 'baseborn', 'based', 'baseless', 'basic', 'bathyal', 'battleful', 'battlemented', 'batty', 'batwing', 'bias', 'calm', 'capable', 'careful', 'careless', 'caring', 'cautious', 'charming', 'cheap', 'cheerful', 'chubby', 'clean', 'clever', 'clumsy', 'cold', 'colorful', 'comfortable', 'concerned', 'confused', 'crowded', 'cruel', 'curious', 'curly', 'cute', 'damaged', 'dangerous', 'dark', 'deep', 'defective', 'delicate', 'delicious', 'depressed', 'determined', 'different', 'dirty', 'disgusting', 'dry', 'dusty', 'daft', 'daily', 'dainty', 'damn', 'damning', 'damp', 'dampish', 'darkling', 'darned', 'dauntless', 'daylong', 'early', 'educated', 'efficient', 'elderly', 'elegant', 'embarrassed', 'empty', 'encouraging', 'enthusiastic', 'excellent', 'exciting', 'expensive', 'fabulous', 'fair', 'faithful', 'famous', 'fancy', 'fantastic', 'fast', 'fearful', 'fearless', 'fertile', 'filthy', 'foolish', 'forgetful', 'friendly', 'funny', 'gentle', 'glamorous', 'glorious', 'gorgeous', 'graceful', 'grateful', 'great', 'greedy', 'green', 'generous', 'gracious', 'genuine', 'grand', 'groovy', 'gutsy', 'gargantuan', 'giddy', 'glistening', 'good', 'golden', 'grouchy', 'grumpy', 'happy', 'humble', 'handsome', 'helpful', 'hilarious', 'healthy', 'hardworking', 'hopeful', 'honest', 'hearty', 'harmonious', 'high-spirited', 'haughty', 'hasty', 'heavy', 'hot', 'horrific', 'hypnotic', 'hypersensitive', 'hyperactive', 'handsome', 'happy', 'harsh', 'healthy', 'heavy', 'helpful', 'hilarious', 'historical', 'horrible', 'hot', 'huge', 'humorous', 'hungry', 'innocent', 'inquisitive', 'intense', 'impressive', 'intelligent', 'interesting', 'incredible', 'irresistible', 'indispensable', 'imaginative', 'impartial', 'ideal', 'immaculate', 'impeccable', 'imperfect', 'imposing', 'impulsive', 'incomparable', 'inconsistent', 'incontrovertible', 'ignorant', 'illegal', 'imaginary', 'impolite', 'important', 'impossible', 'jolly', 'joyful', 'juicy', 'jumpy', 'jovial', 'jaded', 'jazzy', 'jittery', 'jocund', 'jumbled', 'jarring', 'jaunty', 'jingoistic', 'jovial', 'judicious', 'jumpy', 'jovial', 'jocose', 'jittery', 'joyous', 'kind', 'keen', 'knowledgeable', 'kinetic', 'kooky', 'knockout', 'karmic', 'kooky', 'kaleidoscopic', 'kempt', 'kooky', 'kooky', 'kittenish', 'knotty', 'knightly', 'knobby', 'knitting', 'knockdown', 'knuckleheaded', 'knowledgeable', 'magnificent', 'majestic', 'mysterious', 'moody', 'modest', 'merry', 'modern', 'masculine', 'magical', 'melodic', 'mischievous', 'mindful', 'mighty', 'mature', 'mythical', 'mellow', 'multifaceted', 'muscular', 'magnanimous', 'memorable', 'macho', 'massive', 'mean', 'messy', 'nice', 'neat', 'nasty', 'noble', 'nurturing', 'nervous', 'nifty', 'nimble', 'natural', 'notable', 'noisy', 'numerous', 'nutritious', 'nonchalant', 'nostalgic', 'nuclear', 'numbing', 'numinous', 'nurtured', 'obedient', 'obese', 'obnoxious', 'old', 'overconfident', 'optimistic', 'open-minded', 'outstanding', 'original', 'observant', 'obliging', 'odd', 'oily', 'old-fashioned', 'opaque', 'opportunistic', 'optimized', 'organic', 'ornate', 'oscillating', 'outgoing', 'overjoyed', 'overwhelming', 'passionate', 'patient', 'playful', 'pleasant', 'positive', 'powerful', 'precise', 'pretty', 'profound', 'proud', 'pure', 'puzzled', 'peaceful', 'pensive', 'perky', 'petite', 'phenomenal', 'plucky', 'polished', 'popular', 'pink', 'polite', 'poor', 'precious', 'quick', 'quiet', 'qualified', 'quaint', 'querulous', 'quirky', 'quixotic', 'quintessential', 'quivering', 'quizzical', 'rapid', 'rare', 'red', 'remarkable', 'responsible', 'rich', 'romantic', 'royal', 'rude', 'scintillating', 'secretive', 'selfish', 'serious', 'sharp', 'shiny', 'shocking', 'short', 'shy', 'silly', 'sincere', 'skinny', 'slim', 'slow', 'small', 'soft', 'spicy', 'spiritual', 'splendid', 'strong', 'successful', 'sweet', 'talented', 'tall', 'tense', 'terrible', 'terrific', 'thick', 'thin', 'tiny', 'tactful', 'tailor-made', 'take-charge', 'tangible', 'tasteful', 'tasty', 'teachable', 'teeming', 'tempean', 'temperate', 'tenable', 'tenacious', 'tender', 'tender-hearted', 'terrific', 'testimonial', 'thankful', 'thankworthy', 'therapeutic', 'thorough', 'thoughtful', 'ubiquitous', 'ugly', 'ultimate', 'ultra', 'unabashed', 'unafraid', 'unappealing', 'unassuming', 'unaware', 'unbelievable', 'unbiased', 'uncommon', 'unconditional', 'unconventional', 'unctuous', 'undaunted', 'understated', 'unequivocal', 'unforgettable', 'unhappy', 'unique', 'untidy', 'upset', 'valiant', 'vibrant', 'vigorous', 'vivacious', 'versatile', 'vast', 'vengeful', 'venomous', 'viable', 'vigilant', 'vindictive', 'violent', 'virtuous', 'vocal', 'volatile', 'voluptuous', 'voracious', 'vulnerable', 'vulgar', 'venerated', 'victorious', 'warm', 'witty', 'wise', 'wonderful', 'wondrous', 'wild', 'wealthy', 'whimsical', 'wavy', 'weary', 'weak', 'wicked', 'well-behaved', 'well-groomed', 'well-mannered', 'wholesome', 'wide', 'willful', 'winsome', 'worried', 'yearning', 'yellow', 'yielding', 'young', 'youthful', 'yearly', 'yearlong', 'yummy', 'yawning', 'yare', 'yester', 'yestern', 'yielded', 'yielding', 'yippy', 'yummylicious', 'yucky', 'yowling', 'yawningly', 'yarely', 'zany', 'zappy', 'zestful', 'zesty', 'zippy', 'zen', 'zonal', 'zonalized', 'zoological', 'zonalistic', 'zonked', 'zealous', 'sexy', 'slutty']
clip_text_features_dict = {}
text_features_dict = {}
image_features_cache = {}

def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    # 將inputs轉換為fp16
    inputs["pixel_values"] = inputs["pixel_values"].half()
    
    with torch.cuda.amp.autocast():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    if task_prompt == '<DENSE_REGION_CAPTION>':
        dense_labels = parsed_answer['<DENSE_REGION_CAPTION>']['labels']
        caption = ', '.join([label for label in dense_labels if label.count(' ') > 1])
    elif task_prompt == '<CAPTION_TO_PHRASE_GROUNDING>':
        bboxes = parsed_answer['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
        num_persons = len(bboxes)
        num_persons_word = p.number_to_words(num_persons)
        if num_persons > 1:
            caption = f'{num_persons_word} person' if num_persons < 6 else '6+persons'
        else:
            caption = ''
    else:
        caption = parsed_answer[task_prompt].strip().replace('\n', ' ').replace(', ', ' ').replace('.', ',')
        if caption[-1]==',':
            return caption[:-1]
    return caption

def generate_special_text(image_path, args, features=None, chars=None):
    """
    根據 features, image_path 和 parent_folder 生成 special_text。
    """
    def has_reverse_name(name_set, name):
        """
        檢查 name_set 中是否存在 name 的相反名稱（中間有一個空格）。
        """
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
    if args.folder_name and "_" in parent_folder and parent_folder.split("_")[0].isdigit():
        if not args.not_char:
            chartag_from_folder = parent_folder.split('_')[1].replace('_', ' ').strip()
            chartags.add(chartag_from_folder)
        else:
            concept_tag = f"{parent_folder.split('_')[1].replace('_', ' ').strip(), } is main concept of image, "
            
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
        return f"{chartag_from_folder} is the character in the image", ', '.join(chartags), boorutag
    
    if not chartag_from_folder and features and chartags and "solo" in features:
        return f"{concept_tag}{' '.join(chartags)} is the character in the image" if chartags else "", ', '.join(chartags), boorutag
        
    if chartags and len(chartags) <= 3:
        return f'{concept_tag}the characters in this image are {" and ".join(chartags)}', ', '.join(chartags), boorutag
    
    return f'{concept_tag}{chartag_from_folder} and lots of characters in this image', ', '.join(chartags), boorutag

def calculate_best_labels(image, long_caption, short_caption, image_path):  
    labels = [label for label in long_caption.split(", ") + short_caption.split(", ") if label.strip()]
    image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features = F.normalize(image_features, dim=-1) 
    # 计算所有标签的 text_features
    
    clip_scores = []
    for label in clip_labels:
        with torch.no_grad():
            logits_per_image = (image_features @ clip_text_features_dict[label].T).item()
            clip_scores.append((label, logits_per_image))
    
    clip_scores.sort(key=lambda x: x[1], reverse=True)
    top_clip_labels = [clip_scores[0][0], clip_scores[1][0], clip_scores[2][0]]
    labels = labels + top_clip_labels
    for label in labels:
        if label not in text_features_dict:
            text_tensor = longclip.tokenize([label]).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text_tensor)
                text_features = F.normalize(text_features, dim=-1)
            text_features_dict[label] = text_features

    final_labels = []
    best_score = float('-inf')
    best_labels = ""
    score_improvements = []
    
    initial_score = None
    final_score = None
    thresholds = [0.1, 0.4, 0.9]
    selected_labels = [""] * 5
    
    while True:
        improved = False
        for label in labels:
            combined_text = ', '.join(final_labels + [label])
            if args.use_norm:
                combined_text_features = torch.cat([text_features_dict[l] for l in final_labels + [label]], dim=0).mean(dim=0, keepdim=True)
                combined_text_features = F.normalize(combined_text_features, dim=-1)
            else:    
                combined_text_features = torch.cat([text_features_dict[l] for l in final_labels + [label]], dim=0).sum(dim=0, keepdim=True)
            with torch.no_grad():
                logits_per_image = (image_features @ combined_text_features.T).item()
            if logits_per_image > best_score:
                best_score = logits_per_image
                best_labels = combined_text
                improved = True
        if improved:
            best_label = best_labels.split(", ")[-1]
            final_labels.append(best_label)
            if best_label in labels:
                labels.remove(best_label)
            if initial_score is None:
                initial_score = best_score
            final_score = best_score
            score_improvements.append((best_score, best_labels))
        else:
            break

    # 计算分数提升
    if initial_score is not None and final_score is not None:
        for score, labels in score_improvements:
            improvement = (score - initial_score) / (final_score - initial_score)
            for i, threshold in enumerate(thresholds):
                if selected_labels[i] == "" and improvement >= threshold:
                    selected_labels[i] = labels

    combined_text_features = F.normalize(combined_text_features, dim=-1)
    with torch.no_grad():
        final_score = (image_features @ combined_text_features.T).item()
    return selected_labels, final_score

def build_folder_chartag(text, folder_chartag):
    """
    构建folder_chartag字典
    输入: 字符串text, 集合chartags
    输出: folder_chartag字典
    """
    tags = [tag.strip() for tag in text.split(',')]
    folder_chartag = {} if folder_chartag is None else folder_chartag
    
    for tag in tags:
        if tag in chartags:
            if tag in folder_chartag:
                folder_chartag[tag] += 1
            else:
                folder_chartag[tag] = 1
                
    return folder_chartag

def process_image(image_path, folder_chartag, args):
    """
    處理單個圖片，獲取標籤並存儲。修改以支持多進程數據傳遞。
    """

    def resize_image(image_path, max_size=640):
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

    def process_features(features: dict) -> (dict, str):
        """
        處理features字典，移除指定模式的鍵值對並生成keep_tags字串。
        
        參數:
        features (dict): 包含特徵的字典。

        返回:
        (dict, str): 返回處理後的features字典和keep_tags字串。
        """
        def process_people(keep_tags_set):
            boygirl_tags = {tag for tag in keep_tags_set if tag in {'multiple_girls', '1girl', 'multiple_boys', '1boy'}}
            if boygirl_tags:
                combined_tag = ' and '.join(boygirl_tags)
                keep_tags_set.add(combined_tag)
                keep_tags_set.difference_update(boygirl_tags)            
            return keep_tags_set
        
        patterns_to_keep = [
            r'^anime.*$', r'^monochrome$', r'^.*background$', r'^comic$', r'^greyscale$', r'^sketch$' 
            r'^.*censor.*$', r'^.*_name$', r'^signature$', r'^.*_username$', r'^.*text.*$', 
            r'^.*_bubble$', r'^multiple_views$', r'^.*blurry.*$', r'^.*koma$', r'^watermark$', 
            r'^traditional_media$', r'^parody$', r'^.*cover$', r'^.*_theme$', r'^.*realistic$', 
            r'^oekaki$', r'^3d$', r'^.*chart$', r'^letterboxed$', r'^variations$', r'^.*mosaic.*$', 
            r'^omake$', r'^column.*$', r'^.*_(medium)$', r'^manga$', r'^lineart$', r'^.*logo$', 
            #r'^1girl$', r'^1boy$', r'^multiple_boys$', r'^multiple_girls$', r'^69$', r'^absolutely_everyone$', r'^after_kiss$', r'^age_comparison$', r'^age_difference$', r'^age_progression$', r'^angel_and_devil$', r'^anilingus$', r'^ankle_grab$', r'^anti-aircraft$', r'^armpit_sex$', r'^arms_around_neck$', r'^arms_around_waist$', r'^arm_around_back$', r'^arm_around_neck$', r'^arm_around_shoulder$', r'^arm_around_waist$', r'^arm_held_back$', r'^arm_hug$', r'^ass-to-ass$', r'^asymmetrical_docking$', r'^back-to-back$', r'^band$', r'^behind_another$', r'^black_vs_white$', r'^bound_together$', r'^boy_on_top$', r'^boy_sandwich$', r'^breastfeeding$', r'^breasts_on_head$', r'^breast_envy$', r'^grabbing_another\'s_breast$', r'^breast_smother$', r'^breast_sucking$', r'^buttjob$', r'^caressing_testicles$', r'^carrying_person$', r'^chart$', r'^chasing$', r'^cheating_\(relationship\)$', r'^cheek-to-cheek$', r'^chikan$', r'^child_carry$', r'^child_on_child$', r'^circle_formation$', r'^clog_sandals$', r'^clone$', r'^clothed_female_nude_female$', r'^clothed_female_nude_male$', r'^clothed_male_nude_female$', r'^clothed_sex$', r'^coffee_cup$', r'^collage$', r'^colored_text$', r'^column_lineup$', r'^comforting$', r'^cooperative_fellatio$', r'^cooperative_paizuri$', r'^copyright$', r'^costume_switch$', r'^couple$', r'^cousins$', r'^covering_another\'s_eyes$', r'^covering_another\'s_mouth$', r'^covering_mouth$', r'^cowgirl_position$', r'^cross-section$', r'^cuddling$', r'^cum_in_nose$', r'^cum_overflow$', r'^cunnilingus$', r'^cute_$', r'^dark_penis$', r'^deepthroat$', r'^deep_penetration$', r'^disembodied_limb$', r'^disembodied_penis$', r'^doggystyle$', r'^double_handjob$', r'^dressing_another$', r'^dual_persona$', r'^duckling$', r'^duel$', r'^ear_biting$', r'^ejaculating_while_penetrated$', r'^ejaculation$', r'^emotionless_sex$', r'^everyone$', r'^evolutionary_line$', r'^expression_chart$', r'^eye_contact$', r'^face-to-face$', r'^facepalm$', r'^face_to_breasts$', r'^facing_another$', r'^fellatio$', r'^female_child$', r'^femdom$', r'^fff_threesome$', r'^ffm_threesome$', r'^fighting$', r'^finger_biting$', r'^finger_in_another\'s_mouth$', r'^finger_to_another\'s_mouth$', r'^flashback$', r'^flat_chest_grab$', r'^fleeing$', r'^footjob$', r'^foot_worship$', r'^forehead-to-forehead$', r'^french_kiss$', r'^friends$', r'^frilled_swimsuit$', r'^frottage$', r'^full_nelson$', r'^fume$', r'^furry_with_furry$', r'^furry_with_non-furry$', r'^futa_on_male$', r'^futa_with_female$', r'^futa_with_futa$', r'^futa_with_male$', r'^gangbang$', r'^girl_on_top$', r'^girl_sandwich$', r'^glansjob$', r'^glomp$', r'^gloved_handjob$', r'^grabbing$', r'^grabbing_another\'s_ass$', r'^grabbing_another\'s_breast$', r'^grabbing_another\'s_chin$', r'^grabbing_another\'s_hair$', r'^grabbing_from_behind$', r'^greek_clothes$', r'^griffin_$', r'^grinding$', r'^groom$', r'^groping$', r'^group_hug$', r'^group_picture$', r'^group_sex$', r'^guided_breast_grab$', r'^guided_penetration$', r'^guiding_hand$', r'^hairjob$', r'^handjob$', r'^handshake$', r'^hands_on_another\'s_cheeks$', r'^hands_on_another\'s_chest$', r'^hands_on_another\'s_face$', r'^hands_on_another\'s_head$', r'^hands_on_another\'s_hips$', r'^hands_on_another\'s_shoulders$', r'^hands_on_another\'s_thighs$', r'^hands_on_shoulders$', r'^hand_grab$', r'^hand_in_another\'s_hair$', r'^hand_on_another\'s_arm$', r'^hand_on_another\'s_ass$', r'^hand_on_another\'s_back$', r'^hand_on_another\'s_cheek$', r'^hand_on_another\'s_chest$', r'^hand_on_another\'s_chin$', r'^hand_on_another\'s_ear$', r'^hand_on_another\'s_face$', r'^hand_on_another\'s_hand$', r'^hand_on_another\'s_head$', r'^hand_on_another\'s_hip$', r'^hand_on_another\'s_leg$', r'^hand_on_another\'s_neck$', r'^hand_on_another\'s_shoulder$', r'^hand_on_another\'s_stomach$', r'^hand_on_another\'s_thigh$', r'^hand_on_another\'s_waist$', r'^happy_sex$', r'^harem$', r'^headpat$', r'^heads_together$', r'^head_between_breasts$', r'^head_grab$', r'^head_on_another\'s_shoulder$', r'^head_on_chest$', r'^heart_hands_duo$', r'^heckler_$', r'^height_difference$', r'^hetero$', r'^holding_another\'s_arm$', r'^holding_another\'s_foot$', r'^holding_another\'s_hair$', r'^holding_another\'s_leg$', r'^holding_another\'s_wrist$', r'^holding_hair$', r'^holding_hands$', r'^holding_pokemon$', r'^holomyth$', r'^hoop_piercing$', r'^horn_grab$', r'^hug$', r'^hug_from_behind$', r'^humping$', r'^imminent_fellatio$', r'^imminent_kiss$', r'^imminent_penetration$', r'^imminent_vaginal$', r'^implied_fingering$', r'^implied_futanari$', r'^implied_kiss$', r'^in-franchise_crossover$', r'^incest$', r'^infinity$', r'^instant_loss$', r'^internal_cumshot$', r'^interracial$', r'^interspecies$', r'^invisible_man$', r'^in_the_face$', r'^irrumatio$', r'^jealous$', r'^josou_seme$', r'^just_the_tip$', r'^kabedon$', r'^kanshou_$', r'^kiss$', r'^kissing_cheek$', r'^kissing_forehead$', r'^kissing_hand$', r'^kissing_neck$', r'^kissing_penis$', r'^lap_pillow$', r'^leaning_on_person$', r'^left-to-right_manga$', r'^legwear_under_shorts$', r'^leg_between_thighs$', r'^leg_grab$', r'^leg_lock$', r'^licking_another\'s_face$', r'^licking_armpit$', r'^licking_foot$', r'^licking_nipple$', r'^licking_penis$', r'^lifted_by_another$', r'^lifting_another\'s_clothes$', r'^lifting_person$', r'^light_blue_background$', r'^lineup$', r'^locked_arms$', r'^lolidom$', r'^looking_at_another$', r'^looking_at_penis$', r'^lying_on_lap$', r'^lying_on_person$', r'^massage$', r'^matching_outfits$', r'^matching_outfits$', r'^mating_press$', r'^missionary$', r'^misunderstanding$', r'^mixed-sex_bathing$', r'^mixed_bathing$', r'^mmf_threesome$', r'^mmm_threesome$', r'^mod3_\(girls\'_frontline\)$', r'^molestation$', r'^motherly$', r'^mouse$', r'^mtu_virus$', r'^multiple_4koma$', r'^multiple_boys$', r'^multiple_crossover$', r'^multiple_drawing_challenge$', r'^multiple_girls$', r'^multiple_others$', r'^multiple_penises$', r'^multiple_persona$', r'^multiple_riders$', r'^multiple_views$', r'^multitasking$', r'^mutual_hug$', r'^mutual_masturbation$', r'^netorare$', r'^nipple-to-nipple$', r'^noses_touching$', r'^nursing_handjob$', r'^odd_one_out$', r'^onee-loli$', r'^onee-shota$', r'^onii-shota$', r'^on_person$', r'^oral$', r'^orgy$', r'^out_of_frame$', r'^overflow$', r'^paizuri$', r'^paizuri_under_clothes$', r'^penises_touching$', r'^penis_awe$', r'^penis_grab$', r'^penis_on_ass$', r'^penis_on_face$', r'^penis_size_difference$', r'^people$', r'^perpendicular_paizuri$', r'^person_on_head$', r'^phone_screen$', r'^picture_\(object\)$', r'^piggyback$', r'^pikmin_\(creature\)$', r'^pointing_at_another$', r'^pokemon_on_head$', r'^pokemon_on_shoulder$', r'^pokephilia$', r'^pov_crotch$', r'^pov_hands$', r'^prank$', r'^princess_carry$', r'^print_legwear$', r'^prone_bone$', r'^protecting$', r'^pulled_by_another$', r'^pulling_another\'s_clothes$', r'^pushing$', r'^pushing_away$', r'^reach-around$', r'^remembering$', r'^reverse_cowgirl_position$', r'^reverse_suspended_congress$', r'^reverse_upright_straddle$', r'^rhodes_island_logo$', r'^riding_pokemon$', r'^rotational_symmetry$', r'^rough_sex$', r'^sailor_senshi$', r'^same-sex_bathing$', r'^sandwiched$', r'^see-through_swimsuit$', r'^selfcest$', r'^sequential$', r'^sex$', r'^sextuplets$', r'^sexual_coaching$', r'^sex_from_behind$', r'^shared_bathing$', r'^shared_clothes$', r'^shared_earphones$', r'^shared_food$', r'^shared_object_insertion$', r'^shared_scarf$', r'^shared_speech_bubble$', r'^shared_umbrella$', r'^shimaidon_\(sex\)$', r'^shiny_and_normal$', r'^shoulder_carry$', r'^siblings$', r'^side-by-side$', r'^sisters$', r'^sitting_on_bench$', r'^sitting_on_face$', r'^sitting_on_lap$', r'^sitting_on_person$', r'^sitting_on_shoulder$', r'^size_difference$', r'^slapping$', r'^sleeping_on_person$', r'^sleeve_grab$', r'^sling$', r'^solo_focus$', r'^spitroast$', r'^spitting$', r'^spit_take$', r'^spooning$', r'^square_4koma$', r'^squatting_cowgirl_position$', r'^standing_sex$', r'^starter_pokemon_trio$', r'^stealth_sex$', r'^still_life$', r'^straddling$', r'^straddling_paizuri$', r'^strangling$', r'^strap-on$', r'^surprise_kiss$', r'^surrounded_by_penises$', r'^suspended_congress$', r'^symmetrical_docking$', r'^tail_around_leg$', r'^tail_feathers$', r'^take_your_pick$', r'^teacher_and_student$', r'^teamwork$', r'^team_9$', r'^testicle_grab$', r'^testicle_sucking$', r'^thigh_grab$', r'^thigh_sex$', r'^threesome$', r'^time_paradox$', r'^torso_grab$', r'^tribadism$', r'^triplets$', r'^turnaround$', r'^twincest$', r'^twins$', r'^two-footed_footjob$', r'^two-handed_handjob$', r'^ugly_man$', r'^undressing_another$', r'^upright_straddle$', r'^uterus$', r'^vaginal$', r'^variations$', r'^walk-in$', r'^window_shade$', r'^wrestling$', r'^yaoi$', r'^yuri$', r'^:>=$'
            r'^(from_side|from_behind|from_above|from_below)$', r'^(close_up|dutch_angle|downblouse|downpants|pantyshot|upskirt|atmospheric_perspective|fisheye|panorama|perspective|pov|rotated|sideways|upside_down|vanishing_point|straight-on)$', r'^(face|cowboy_shot|portrait|upper_body|lower_body|feet_out_of_frame|full_body|wide_shot|very_wide_shot|cut_in|cropped_legs|head_out_of_frame|cropped_torso|cropped_arms|cropped_shoulders|profile|group_profile)$', r'^(armpit_focus|ass_focus|back_focus|breast_focus|eye_focus|foot_focus|hand_focus|hip_focus|navel_focus|pectoral_focus|thigh_focus|soft_focus|solo_focus)$'
        ]
        keep_tags_set = set()
        #if 'solo' in features:
        #    patterns_to_keep.extend([r'^holding_.*$', r'^.*grab.*$', r'^.*lift.*$', r'^.*pull$', r'^.*_own_.*$', r'^.*covered.*$', r'^.*_masturbation.*$', r'^.*out.*$', r'^.*_between_.*$'])
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
        
        keep_tags = ', '.join(process_people(keep_tags_set)).rstrip(', ')
        
        return features, keep_tags
    
    tag_file_path = Path(image_path).with_suffix('').with_suffix('.txt')

    # 檢查文件最後修改時間，如果在一周內則略過
    if tag_file_path.exists():
        last_modified_time = datetime.fromtimestamp(tag_file_path.stat().st_mtime)
        if datetime.now() - last_modified_time < timedelta(days=args.continue_caption):
            print(f"Skipping {tag_file_path} as it was modified within the last week.")
            return None, None, 'skipped'   
    try:
        image = resize_image(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 使用 imgutils 獲取圖片等級
        rating, features, chars = get_wd14_tags(image, character_threshold=0.7, general_threshold=0.2682, drop_overlap=True)
        features, keep_tags = process_features(features)
        #features = drop_basic_character_tags(features)
        wd14_caption = tags_to_text(features, use_escape=False, use_spaces=True)
        special_text, chartags, boorutag = generate_special_text(image_path, args, features, chars)
        rating = max(rating, key=rating.get)
        if keep_tags:
            special_text += f", {keep_tags}"    
        if rating:
            replacements = {
                'general': 'safety',
                'sensitive': 'naughty',
                'questionable': 'revealing',
                'explicit': 'porn'
            }
            for old, new in replacements.items():
                rating = rating.replace(old, new)        
            special_text += f", {rating}"
        wd14_caption = wd14_caption + ', ' + boorutag
        more_detailed_caption = run_example('<MORE_DETAILED_CAPTION>', image) 
        clip_caption = []
        clip_caption, final_score = calculate_best_labels(image, more_detailed_caption, wd14_caption, image_path)
        folder_chartag = build_folder_chartag(clip_caption[4], folder_chartag)
        tags_text = (
            f"{special_text}, ___{clip_caption[2]}\n"
            f"{special_text}, ___{clip_caption[1]}\n"
            f"{special_text}, ___{clip_caption[1]}\n"
            f"{special_text}, ___{clip_caption[0]}\n"
            f"{special_text}, ___{clip_caption[0]}"
        )

        with open(tag_file_path, 'w', encoding='utf-8') as f:
            f.write(tags_text.lower()) 
        return folder_chartag, final_score
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")

def drop_chartags_in_folder(folder_path, folder_chartag):
    """
    在指定目录中删除高频chartag
    输入: 目录路径folder_path, folder_chartag字典
    """
    max_count = max(folder_chartag.values())    
    threshold = max_count / 3
    tags_to_drop = {tag for tag, count in folder_chartag.items() if count > threshold}
    
    # 遍历目录中的每个txt文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            for i, line in enumerate(lines):
                new_content = []
                tags = [tag.strip() for tag in line.split(',')]
                for tag in tags:
                    if tag and tag not in tags_to_drop:
                        new_content.append(tag)
                lines[i] = ', '.join(new_content)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write('\n'.join(lines))

def find_and_process_images(directory, args):
    directory = directory.replace('\\', '/')
    extensions = ["*.jpg", "*.png", "*.jpeg", "*.webp", "*.bmp"]
    all_final_scores = []
    for root, dirs, files in os.walk(directory):
        folder_chartag = {}
        image_paths = []
        for ext in extensions:
            for file in files:
                if fnmatch.fnmatchcase(file, ext) or fnmatch.fnmatchcase(file, ext.upper()):
                    image_paths.append(os.path.join(root, file))

        for image_path in tqdm(image_paths, desc=f"處理圖片 {root}"):
            try:
                folder_chartag, final_score = process_image(image_path, folder_chartag, args)  
                all_final_scores.append((image_path, final_score))
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")
        
        if args.drop_chartag and folder_chartag:
            drop_chartags_in_folder(root, folder_chartag)

    if all_final_scores:
        max_score = max(all_final_scores, key=lambda x: x[1])[1]
        min_score = min(all_final_scores, key=lambda x: x[1])[1]

        # 添加accuracy_tag到每个对应的txt文件
        for image_path, final_score in all_final_scores:
            relative_score = (final_score - min_score) / (max_score - min_score)
            if relative_score >= 0.4:
                accuracy_tag = ""
            elif relative_score >= 0.1:
                accuracy_tag = "low accuracy"
            else:
                accuracy_tag = "mess"
            
            tag_file_path = Path(image_path).with_suffix('').with_suffix('.txt')
            if tag_file_path.exists():
                with open(tag_file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                if accuracy_tag:
                    modified_content = content.replace('___', f'{accuracy_tag}, ___')

                # 将修改后的内容写回文件
                with open(tag_file_path, 'w', encoding='utf-8') as file:
                    file.write(modified_content)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="圖片標籤處理腳本")
    parser.add_argument("--folder_name", action="store_true", help="使用目錄名當作角色名")
    parser.add_argument("--drop_chartag", action="store_true", help="自動刪除角色特徵標籤")
    parser.add_argument("--not_char", action="store_true", help="目錄名不是角色")
    parser.add_argument("--use_norm", action="store_true", help="忽略clip文字向量長度，標會較短")
    parser.add_argument("--continue_caption", type=int, default=0, help="忽略n天內打的標")    
    parser.add_argument("directory", type=str, help="處理目錄地址")
    args = parser.parse_args()
    if args.not_char:
        args.folder_name = True
    for label in clip_labels:
        if label not in clip_text_features_dict:
            text_tensor = longclip.tokenize([label]).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text_tensor)
                text_features = F.normalize(text_features, dim=-1)
            clip_text_features_dict[label] = text_features
    text_features_dict = clip_text_features_dict.copy()    
    find_and_process_images(args.directory, args)
