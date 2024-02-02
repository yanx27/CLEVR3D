import json
from collections import defaultdict

exclude_relations = {
    "brighter than",
    "darker than",
    "same material",
    "same texture",
}

exclude_class = ['1', '2', '15']


def process(object='../3DSSG/objects.json', relation='../3DSSG_subset/relationships_validation.json', coords_file='coords.json',
             outfile='3dssg_scenes_val.json'):
    with open(object, 'r', encoding='utf-8') as finObj, \
        open(coords_file, 'r', encoding='utf-8') as fincoord, \
        open(relation, 'r', encoding='utf-8') as finRelation, open(outfile, 'w', encoding='utf-8') as fout:
        objects_lines = json.load(finObj)['scans']
        relations_lines = json.load(finRelation)['scans']
        coords_lines = json.load(fincoord)
        scenes = {}
        for scan in relations_lines: 
            scenes[scan['scan']] = scan

        skips = 0
        for scan in objects_lines:
            try:
                if scenes[scan['scan']]:
                    objects = []
                    for obj in scan['objects']:
                        if obj['rio27'] not in exclude_class:

                            try:
                                if 'rectangular' in obj['attributes']['shape']:
                                    obj['attributes']['shape'].remove('rectangular')
                                    if len(obj['attributes']['shape']) == 0:
                                        del obj['attributes']['shape']

                                if 'wooden' in obj['attributes']['material']:
                                    obj['attributes']['material'].remove('wooden')
                                    if len(obj['attributes']['material']) == 0:
                                        del obj['attributes']['material']
                            except:
                                pass
                            
                            objects.append(obj)
                        
                    scenes[scan['scan']]['objects'] = objects
            except:
                skips += 1
        print('Number of skip: {} (Relation scan_id does not match Object scan_id)'.format(skips))

        # scenes_tmp = scenes
        # scenes = {}

        # ######### add coords to scenes
        # for scan_id, scene in scenes_tmp.items():
        #     try:
        #         coord = coords_lines[scan_id]
        #         for obj in scene['objects']:
        #             obj['3d_coords'] = coord[obj['id']]
        #         scenes[scan_id] = scene
        #     except:
        #         continue

        ######### add relationships
        for scene in list(scenes.values()):
            objects = scene['objects']
            # relations = scene['relationships'][0]  # ! only in SSG_AUG
            relations = scene['relationships']
            object_ids = set(int(x['id']) for x in objects)

            relations_new = defaultdict(list)
            for relation in relations:
                if len(object_ids.intersection(relation[:2])) == 2:
                    relations_new[relation[-1]].append(relation)

            ######## add all relations
            scene['relationships'] = {}
            for k, v in relations_new.items():
                if k in exclude_relations:
                    continue
                scene['relationships'][k] = sort_list(v)

        ##### add info
        scenes = list(scenes.values())
        final_scenes = {}
        final_scenes['info'] = {
            "version": "1.0",
            "license": "Creative Commons Attribution (CC-BY 4.0)",
            "split": "all",
            "date": "09/18/2021"
        }
        # final_scenes['scenes'] = scenes[100:101]
        final_scenes['scenes'] = scenes

        ####### edit information of scene
        for idx, scene in enumerate(final_scenes['scenes']):
            scene['split'] = 'all'
            scene['image_index'] = idx
            scene['image_filename'] = '3dssg_all_' + str(idx) + '.png'

        ######## globle info
        all_colors = defaultdict(int)

        ########## edit object information
        for scene in final_scenes['scenes']:
            objects = scene['objects']
            for idx, obj in enumerate(objects):
                #####
                attributes = obj['attributes']
                shape = attributes.get('shape')
                color = attributes.get('color')
                size = attributes.get('size')
                material = attributes.get('material')

                #######
                if shape:
                    obj['shape'] = shape[0]
                else:
                    obj['shape'] = shape
                if color:
                    obj['color'] = color[0]
                else:
                    obj['color'] = color
                if size:
                    obj['size'] = size[0]
                else:
                    obj['size'] = size
                if material:
                    obj['material'] = material[0]
                else:
                    obj['material'] = material

        ########## outfile
        json.dump(final_scenes, fout, indent=4, ensure_ascii=False)
        print('Number of scenes: %d' % len(final_scenes['scenes']))


def sort_list(lists):
    dicts = defaultdict(list)
    for idx, pair in enumerate(lists):
        dicts[pair[0]].append(pair[1])
    result = list(dicts.values())
    result.append([])
    return dicts


if __name__ == '__main__':
    # process(object='../SSG_AUG/objects.json', relation='../SSG_AUG/relationships.json',
    #         outfile='../question_generation/ssg_aug_scenes.json')
    process()
