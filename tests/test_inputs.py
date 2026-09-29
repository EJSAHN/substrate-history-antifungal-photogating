"""File-safety checks use temporary artificial inputs, not biological evidence."""
import csv,json,sys,tempfile,unittest,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from uvsm.inputs import InputFiles
from uvsm import analysis as a
from uvsm.pipeline import run

class InputChecks(unittest.TestCase):
    def make(self,root):
        (root/'phenotype.csv').write_text('Strain,Chemical\nX,Y\n')
        (root/'fluorescence').mkdir()
        for g in 'ABCDEF':
            for i in range(1,10 if g in 'ABCD' else 5):
                (root/'fluorescence'/f'{g}{i}_F_pixelSpectra.csv').write_text('row,col,500.0\n1,1,1\n')
    def test_dedup_identical_export(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'inputs';root.mkdir();self.make(root)
            extra=root/'fluorescence/copy';extra.mkdir()
            (extra/'A1_F_pixelSpectra.csv').write_bytes((root/'fluorescence/A1_F_pixelSpectra.csv').read_bytes())
            f=InputFiles(root,Path(td)/'work')
            self.assertEqual(len(f.pixels()),44);f.verify_unchanged()
    def test_conflicting_export_stops(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.make(root);extra=root/'fluorescence/copy';extra.mkdir()
            (extra/'A1_F_pixelSpectra.csv').write_text('row,col,500.0\n1,1,9\n')
            with self.assertRaisesRegex(ValueError,'Conflicting'):InputFiles(root,root/'work')
    def test_both_folder_and_zip_stops(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.make(root);(root/'HSI.zip').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError,'Ambiguous'):InputFiles(root,root/'work')
    def test_missing_id_stops(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.make(root);(root/'fluorescence/A1_F_pixelSpectra.csv').unlink()
            with self.assertRaisesRegex(ValueError,'ID mismatch'):InputFiles(root,root/'work')
    def test_hash_manifest_stops_on_change(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.make(root)
            (root/'input_manifest.json').write_text(json.dumps({'files':[]}))
            with self.assertRaisesRegex(ValueError,'hashes differ'):InputFiles(root,root/'work')
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'out';out.mkdir();(out/'KEEP').write_text('original')
            with self.assertRaises(FileExistsError):run(Path(td)/'in',out)
            self.assertEqual((out/'KEEP').read_text(),'original')
    def test_no_input_output_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,'separate'):run(Path(td),Path(td)/'out')

if __name__=='__main__':unittest.main(verbosity=2)
