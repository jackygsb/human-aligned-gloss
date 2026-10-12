from pathlib import Path
import re,json
B=Path(__file__).resolve().parents[1]
O=B/'output/NMI_English_Manuscript_v10'
s=(O/'main_user_20261007.tex').read_text(encoding='utf-8')
blocks=[]
pattern=re.compile(r'\{\s*\\color\{red\}')
while (m:=pattern.search(s)):
 start=m.start();depth=0;end=None
 for j in range(start,len(s)):
  if s[j]=='{' and (j==0 or s[j-1]!='\\'):depth+=1
  elif s[j]=='}' and (j==0 or s[j-1]!='\\'):
   depth-=1
   if depth==0:end=j+1;break
 if end is None:raise ValueError('Unbalanced red block')
 blocks.append(s[start:end]);s=s[:start]+s[end:]
# Remove headings stranded by deletion of their obsolete paragraphs.
s=re.sub(r'\\(subsection|section)\{[^}]+\}\s*(?=\\\1\{)', '',s)
(O/'audit/user_clean_nonred.tex').write_text(s,encoding='utf-8')
(O/'audit/removed_red_blocks.txt').write_text('\n\n--- REMOVED OLD BLOCK ---\n\n'.join(blocks),encoding='utf-8')
def wc(t):
 t=re.sub(r'\\(?:cite|ref|label)\{[^}]+\}','',t);t=re.sub(r'\\[A-Za-z]+',' ',t)
 return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*",t))
report={'red_blocks_removed':len(blocks),'clean_main_words':wc(s.split('\\maketitle')[1].split('\\section{Methods}')[0]),'clean_methods_words':wc(s.split('\\section{Methods}')[1].split('\\section*{Data availability}')[0])}
(O/'audit/initial_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
