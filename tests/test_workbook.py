"""Export-only regression: Unicode XML must remain valid after formula caching."""
from pathlib import Path
import sys,tempfile,unittest,zipfile,xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from uvsm.workbook import write_portable
class WorkbookChecks(unittest.TestCase):
 def test_unicode_formula_xml_and_cache(self):
  spec={'sheets':[{'name':'Check','columns':3,'data_rows':1,
   'values':[['Encoding check',None,None],['Mean \u2212 control; \u0394 and mm\u00b2',None,None],[None]*3,['x','y','difference'],[4.0,1.0,3.0]],
   'formulas':[('C5','=A5-B5')]}]}
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'check.xlsx';write_portable(spec,p)
   with zipfile.ZipFile(p) as z:
    for member in z.namelist():
     if member.endswith('.xml'):ET.fromstring(z.read(member))
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    root=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    self.assertEqual(root.find('.//s:c[@r="C5"]/s:v',ns).text,'3.0')
if __name__=='__main__':unittest.main()
