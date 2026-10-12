from pathlib import Path
import requests,json,time,re
O=Path(__file__).resolve().parents[1]/'mllm_material/nmi_revision_v8/source'
items={
'chadwick2015':'10.1016/j.visres.2014.10.026',
'vangorp2007':'10.1145/1275808.1276473',
'olkkonen2011':'10.1068/i0480',
'doerschner2010':'10.1167/10.4.8',
'marlow2013':'Generative constraints on image cues for perceived gloss',
'marlow2011':'The role of brightness and orientation congruence in the perception of surface gloss',
'fleming2004':'Specular reflections and the perception of shape',
'doerschner2011':'Visual motion and the perception of surface material',
'sharan2014':'Efficiency and speed of material categorization in real-world images',
'wiebel2013':'The speed and accuracy of material recognition in natural images',
'fleming2013':'Perceptual qualities and material classes',
'nishida1998':'Use of image-based information in judgments of surface-reflectance properties',
'mahner2025':'10.1038/s42256-025-01041-7',
'muttenthaler2025':'10.1038/s41586-025-09631-6',
'hebart2019':'10.1371/journal.pone.0223792',
'hebart2023':'THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior',
'peterson2018':'Evaluating and improving the correspondence between deep neural networks and human representations',
'geirhos2021':'10.48550/arXiv.2106.07411',
'bau2017':'Network Dissection: Quantifying Interpretability of Deep Visual Representations',
'bau2020':'10.1073/pnas.1907375117',
'kriegeskorte2016':'10.1016/j.neuroimage.2015.12.012',
'owen2012':'10.1214/12-AOAS547',
'krakauer2017':'10.1016/j.neuron.2016.12.041',
}
cache=O/'provenance/v8_reference_metadata.json';meta=json.loads(cache.read_text('utf-8')) if cache.exists() else {}
if meta.get('hebart2019',{}).get('DOI')=='10.1101/545954':meta.pop('hebart2019')
s=requests.Session();s.headers['User-Agent']='MaterialPerceptionManuscript/1.0 (bibliographic verification)'
for key,query in items.items():
 if key in meta or 'arXiv' in query:continue
 for attempt in range(4):
  try:
   if query.startswith('10.'):r=s.get('https://api.crossref.org/works/'+query,timeout=45);r.raise_for_status();m=r.json()['message']
   else:r=s.get('https://api.crossref.org/works',params={'query.title':query,'rows':1},timeout=45);r.raise_for_status();m=r.json()['message']['items'][0]
   meta[key]=m;cache.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(key,m['title'][0],m.get('DOI'),flush=True);break
  except Exception as e:
   print(key,str(e)[:80],flush=True);time.sleep(2+attempt*2)
def esc(x):return re.sub('<[^>]*>','',str(x)).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%').replace('–','--')
entries=[]
for key,m in meta.items():
 fields={'author':' and '.join(a.get('family','')+', '+a.get('given','') for a in m.get('author',[])),'title':m['title'][0],'journal':m.get('container-title',[''])[0],'year':m.get('published',m['issued'])['date-parts'][0][0],'volume':m.get('volume',''),'pages':m.get('page',m.get('article-number','')),'doi':m['DOI']}
 entries.append('@article{'+key+',\n'+',\n'.join(' '+k+' = {'+esc(v)+'}' for k,v in fields.items() if v!='')+'\n}')
(O/'new_references.bib').write_text('\n\n'.join(entries),encoding='utf-8')
