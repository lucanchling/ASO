from Methode.Methode import Methode
import webbrowser
import glob 

class CBCT(Methode):
    def __init__(self, widget):
        super().__init__(widget)

    def NumberScan(self, scan_folder: str):
        scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
        dic = super().search(scan_folder,scan_extension)
        lenscan=0
        for key in scan_extension:
            lenscan+=len(dic[key])
        return lenscan
        
    def TestScan(self, scan_folder: str):
        out = None
        if self.NumberScan(scan_folder) == 0 :
            out = 'Folder selected do not contain scans'
        return out

    def TestReference(self, ref_folder: str):
        list = glob.glob(ref_folder+'/*.nii.gz')
        out = None
        if len(list) == 0:
            out = 'Please choose a folder with json file'
        elif len(list)>2:
            out = 'Too many json file '
        return out

    def TestCheckbox(self,dic_checkbox):
        list_landmark = self.__CheckboxisChecked(dic_checkbox)
        out = None
        if len(list_landmark.split(','))< 3:
             out = 'Give minimum 3 landmark'
        return out    

    def DownloadRef(self):
        webbrowser.open('https://google.com')

    def TestProcess(self, **kwargs) -> str:
        return super().TestProcess(kwargs)
    
    def Process(self, **kwargs):
        return super().Process(kwargs)

    def DicLandmark(self):
        dic = {'Head':
                {'Cranial Base' : ['Ba', 'S', 'N', 'RPo', 'LPo', 'RFZyg', 'LFZyg', 'C2', 'C3', 'C4'],

                'Upper' : ['RInfOr', 'LInfOr', 'LMZyg', 'RPF', 'LPF', 'PNS', 'ANS', 'A', 'UR3O', 'UR1O', 'UL3O', 'UR6DB', 'UR6MB', 'UL6MB', 'UL6DB', 'IF', 'ROr', 'LOr', 'RMZyg', 'RNC', 'LNC', 'UR7O', 'UR5O', 'UR4O', 'UR2O', 'UL1O', 'UL2O', 'UL4O', 'UL5O', 'UL7O', 'UL7R', 'UL5R', 'UL4R', 'UL2R', 'UL1R', 'UR2R', 'UR4R', 'UR5R', 'UR7R', 'UR6MP', 'UL6MP', 'UL6R', 'UR6R', 'UR6O', 'UL6O', 'UL3R', 'UR3R', 'UR1R'],

                'Lower' : ['RCo', 'RGo', 'Me', 'Gn', 'Pog', 'PogL', 'B', 'LGo', 'LCo', 'LR1O', 'LL6MB', 'LL6DB', 'LR6MB', 'LR6DB', 'LAF', 'LAE', 'RAF', 'RAE', 'LMCo', 'LLCo', 'RMCo', 'RLCo', 'RMeF', 'LMeF', 'RSig', 'RPRa', 'RARa', 'LSig', 'LARa', 'LPRa', 'LR7R', 'LR5R', 'LR4R', 'LR3R', 'LL3R', 'LL4R', 'LL5R', 'LL7R', 'LL7O', 'LL5O', 'LL4O', 'LL3O', 'LL2O', 'LL1O', 'LR2O', 'LR3O', 'LR4O', 'LR5O', 'LR7O', 'LL6R', 'LR6R', 'LL6O', 'LR6O', 'LR1R', 'LL1R', 'LL2R', 'LR2R'],

                'Impacted Canine' : ['UR3OIP','UL3OIP','UR3RIP','UL3RIP'],

                }}

        return dic

    def ListLandmark(self):
        return super().ListLandmark()
        
    def existsLandmark(self,folderpath,reference_folder):

        return None
        
    def Sugest(self):
        return ['Ba','S','N','RPo','LPo','ROr','LOr']


    def __CheckboxisChecked(self,diccheckbox : dict):
        out=''
        if not len(diccheckbox) == 0:
            for checkboxs in diccheckbox.values():
                for checkbox in checkboxs:
                    if checkbox.isChecked():
                        out+=f'{checkbox.text},'
            while out[0]==',':
                out = out[1:]
            before = None
            for i, letter in enumerate(out):
                if before==',' and letter==',':
                    out = out[:i]+out[i+1:]
                before = letter

            out=out[:-1]
        print('out',out)
        return out