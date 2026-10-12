from pathlib import Path
import json,re
B=Path(__file__).resolve().parents[1];S=B/'mllm_material/nmi_manuscript_zh_v1/source'
meta=json.loads((S/'provenance/bibliography_metadata.json').read_text('utf-8'))
meta.pop('marlow2012',None)
def esc(s):
 s=re.sub('<[^>]+>','',str(s));return s.replace('&',r'\&').replace('_',r'\_').replace('%',r'\%').replace('–','--').replace('—','---').replace('�C','--')
entries=[]
for k,m in meta.items():
 authors=' and '.join(a.get('family','')+', '+a.get('given','') for a in m.get('author',[]))
 year=m.get('published',m.get('issued'))['date-parts'][0][0]
 fields={'author':authors,'title':m['title'][0],'journal':m.get('container-title',[''])[0],'year':year,'volume':m.get('volume',''),'number':m.get('issue',''),'pages':m.get('page',''),'doi':m['DOI']}
 if k=='rsa2008':fields['title']='Representational similarity analysis -- connecting the branches of systems neuroscience'
 entries.append('@article{'+k+',\n'+',\n'.join('  '+f+' = {'+esc(v)+'}' for f,v in fields.items() if v!='')+'\n}')
extra=[
('marlow2012','Marlow, Phillip J. and Kim, Juno and Anderson, Barton L.','The perception and misperception of specular surface reflectance','2012','Current Biology','10.1016/j.cub.2012.08.009','22','1909--1913',''),
('internvl2025','Wang, Weiyun and Gao, Zhangwei and Gu, Lixin and others','InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency','2025','arXiv','10.48550/arXiv.2508.18265','','','https://arxiv.org/abs/2508.18265'),
('qwen2025','Bai, Shuai and Cai, Yuxuan and Chen, Ruizhe and others','Qwen3-VL Technical Report','2025','arXiv','10.48550/arXiv.2511.21631','','','https://arxiv.org/abs/2511.21631'),
('clip2021','Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others','Learning Transferable Visual Models From Natural Language Supervision','2021','Proceedings of Machine Learning Research','','139','8748--8763','https://proceedings.mlr.press/v139/radford21a.html'),
('vit2020','Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others','An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale','2020','arXiv','10.48550/arXiv.2010.11929','','','https://arxiv.org/abs/2010.11929'),
('llava2023','Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae','Visual Instruction Tuning','2023','arXiv','10.48550/arXiv.2304.08485','','','https://arxiv.org/abs/2304.08485'),
('probe2016','Alain, Guillaume and Bengio, Yoshua','Understanding intermediate layers using linear classifier probes','2016','arXiv','10.48550/arXiv.1610.01644','','','https://arxiv.org/abs/1610.01644'),
('hewitt2019','Hewitt, John and Liang, Percy','Designing and Interpreting Probes with Control Tasks','2019','Proceedings of EMNLP-IJCNLP','10.18653/v1/D19-1275','','2733--2743','https://aclanthology.org/D19-1275/'),
('tcav2018','Kim, Been and Wattenberg, Martin and Gilmer, Justin and Cai, Carrie and Wexler, James and Viegas, Fernanda and Sayres, Rory','Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)','2018','Proceedings of Machine Learning Research','','80','2668--2677','https://proceedings.mlr.press/v80/kim18d.html'),
('adebayo2018','Adebayo, Julius and Gilmer, Justin and Muelly, Michael and Goodfellow, Ian and Hardt, Moritz and Kim, Been','Sanity Checks for Saliency Maps','2018','Advances in Neural Information Processing Systems','','31','','https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html'),
('jain2019','Jain, Sarthak and Wallace, Byron C.','Attention is not Explanation','2019','Proceedings of NAACL-HLT','10.18653/v1/N19-1357','','3543--3556','https://aclanthology.org/N19-1357/'),
('internvlcard','{OpenGVLab}','InternVL3.5-4B: official model card','2025','Hugging Face model repository','','','','https://huggingface.co/OpenGVLab/InternVL3_5-4B'),
('qwencard','{Qwen Team}','Qwen3-VL-4B-Instruct: official model card','2025','Hugging Face model repository','','','','https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct'),
]
for k,a,t,y,j,d,v,p,u in extra:
 fields={'author':a,'title':t,'year':y,'journal':j,'doi':d,'volume':v,'pages':p,'url':u}
 entries.append('@article{'+k+',\n'+',\n'.join('  '+f+' = {'+(v if f=='url' else esc(v))+'}' for f,v in fields.items() if v)+'\n}')
(S/'references.bib').write_text('\n\n'.join(entries)+'\n','utf-8')
print(len(entries),'verified references')
