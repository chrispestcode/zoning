import torch
import pickle
from collections import defaultdict
from loguru import logger
from utils.image_utils import *

def process_forms(document):
    groups = [group for group in document.groups if 'form_area' in group.label]
    
    if not groups:
        return []

    forms = defaultdict(dict)
    NUM_REDUNDANT_TEXT_GROUP = 2
    for group_i, group in enumerate(groups):
        page_no = ''
        for i, child in enumerate(group.children):
            try:
                # sample child.cref ==> '#/texts/317'
                substr = child.cref.split('/')
                idx = (int) (substr[-1])
                #TODO: Incorrect form context  captured ==> renders overlapping text group capturing below inconsistent
                text_item = document.texts[idx]
                page_no = text_item.prov[0].page_no
                if 'form_text' not in forms[page_no]:
                    forms[page_no]['form_text'] = []
                if i == 0:
                    ''' Add overlapping group items from previous group'''
                    for j in range(1, NUM_REDUNDANT_TEXT_GROUP+1, 1):
                        forms[page_no]['form_text'].append(document.texts[idx-j].text) if idx-j >= 0 else ""
                elif i == len(group.children)-1:
                    ''' Add overlapping group items from next group'''
                    for j in range(1, NUM_REDUNDANT_TEXT_GROUP+1,1):
                        forms[page_no]['form_text'].append(document.texts[idx+j].text) if idx+j < len(document.texts) else ""
                forms[page_no]['form_text'].append(text_item.text)
            except Exception as e:
                logger.error(f'Child {i} in Group {group_i} could not be extracted')
                return {}
    return [{
        'form_text_full': ';'.join(v['form_text']),
        'page_no': k,
        'page_base64': document.pages[k].image.uri.encoded_string()
        } for k,v in forms.items()]

def process_images(document, device, blip_processor=None, blip_model=None):
    MIN_WIDTH = 100
    MIN_HEIGHT = 100
    filtered_images = [document_img for document_img in document.pictures] # if document_img.image.size.height >= MIN_HEIGHT and document_img.image.size.width >= MIN_WIDTH ]
    images = [{

                'ref': img_data.self_ref,
                'mime_type': img_data.image.mimetype,
                'height': img_data.image.size.height,
                'width': img_data.image.size.width,
                'dpi': img_data.image.dpi,
                'base64': img_data.image.uri.path,
                'page_no': img_data.prov[0].page_no
                } for img_data in filtered_images]

    if not images:
        return []

    # Generate alt text using BLIP-2
    # for item in images:
    #     if 'base64' not in item:
    #         continue
    #     image = decode_image(item['base64'])
    #     inputs = blip_processor(image, return_tensors="pt").to(device)
    #     ''' document_data['document'].pictures[1].image '''
    #     ''' ImageRef(mimetype='image/png', dpi=144, size=Size(width=1224.0, height=820.0), uri=AnyUrl('data:image/png;base64, <encoding...'''
    #     ''' to get encoding: document_data['document'].pictures[1].image.uri.path <== bare string'''
    #     try:
    #         with torch.no_grad():
    #             outputs = blip_model.generate(
    #                 **inputs,
    #                 max_length=100,
    #                 num_beams=5,
    #                 min_length=10,
    #                 top_p=0.9,
    #                 repetition_penalty =1.5,
    #                 length_penalty=1.0,
    #                 temperature=0.7,
    #                 use_cache=False
    #             )
            
    #         alt_text = blip_processor.decode(outputs[0], skip_special_tokens=True)
    #         item['alt_text'] = alt_text
    #         del inputs
    #         del outputs
    #         image.close()
    #     except Exception as e:
    #         logger.error(f"Error encountered while generating alt text for ref {item.ref}")
    #        return []
    return images

def process_tables(document, device, blip_processor=None, blip_model=None):
    tables_data = [{
            'ref': table.get_ref().cref,
            'bbox': get_bounding_box(table),
            'num_rows': table.data.num_rows,
            'num_cols': table.data.num_cols,
            'dataframe': pickle.dumps(table.export_to_dataframe()),
            'page_no': table.prov[0].page_no,
            'base64': encode_image(table.get_image(document))

        } for table in document.tables]

    prompt =  "The table has rows and columns describing "
    # for table in tables_data:
    #     if 'base64' not in table:
    #         continue
        # inputs = blip_processor(decode_image(table['base64']), text=prompt, return_tensors="pt").to(device)
        # try:
        #     with torch.no_grad():
        #         outputs = blip_model.generate(
        #             **inputs,
        #             max_length=150,
        #             num_beams=5,
        #             min_length=20,
        #             use_cache=False,
        #             do_sample=True
        #         )
        
        #     alt_text = blip_processor.decode(outputs[0], skip_special_tokens=True)
        #     table['alt_text'] = alt_text
        #     del inputs
        #     del outputs
        # except Exception as e:
        #     logger.error("Error encountered while generating caption for table", e)
        #     return []
    return tables_data