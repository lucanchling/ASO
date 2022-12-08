from Methode.Methode import Methode
import webbrowser
import os 
import slicer
import json

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
        
    def PatientScanLandmark(self, dic, scan_extension, lm_extension):
        patients = {}

        for extension,files in dic.items():
            for file in files:
                file_name = os.path.basename(file).split(".")[0]
                patient = file_name.split('_scan')[0].split('_Scanreg')[0].split('_lm')[0]

                if patient not in patients.keys():
                    patients[patient] = {"dir": os.path.dirname(file),"lmrk":[]}
                if extension in scan_extension:
                    patients[patient]["scan"] = file
                if extension in lm_extension:
                    patients[patient]["lmrk"].append(file)

        return patients
    
    def TestScan(self, scan_folder: str):
        out = ''
        scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
        lm_extension = [".json"]

        if self.NumberScan(scan_folder) == 0 :
            return 'The selected folder must contain scans'
        
        dic = super().search(scan_folder,scan_extension,lm_extension)

        patients = self.PatientScanLandmark(dic,scan_extension,lm_extension)

        for patient,data in patients.items():
            if "scan" not in data.keys():
                out += "Missing scan for patient : {}\nat {}\n".format(patient,data["dir"])
            if len(data['lmrk']) == 0:
                out += "Missing landmark for patient : {}\nat {}\n".format(patient,data["dir"])
        
        if out == '':   # If no errors
            out = None
        return out

    def TestReference(self, ref_folder: str):
        out = None
        scan_extension = [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]
        lm_extension = [".json"]

        if self.NumberScan(ref_folder) == 0 :
            out = 'The selected folder must contain scans'

        if self.NumberScan(ref_folder) > 1 :
            out = 'The selected folder must contain only 1 case'
        
        return out

    def TestCheckbox(self,dic_checkbox):
        list_landmark = self.__CheckboxisChecked(dic_checkbox)
        out = None
        if len(list_landmark) < 3:
             out = 'Select a minimum of 3 landmarks\n'
        return out    

    def TestProcess(self, **kwargs) -> str:
        out=''

                
        testcheckbox = self.TestCheckbox(kwargs['dic_checkbox'])
        if testcheckbox is not None:
            out+=testcheckbox

        if kwargs['input_folder'] == '':
            out+= 'Select an input folder\n'

        if kwargs['gold_folder'] == '':
            out+= 'Select a reference folder\n'

        if kwargs['folder_output'] == '':
            out+= 'Select an output folder\n'

        if kwargs['add_in_namefile']== '':
            out += 'Select an extension for output files\n'

        if out == '':
            out = None

        return out

    def Process(self, **kwargs):
        list_lmrk = self.__CheckboxisChecked(kwargs['dic_checkbox'])
        list_lmrk_str = ''
        for lm in list_lmrk:
            list_lmrk_str+=lm+','
        
        parameter= {'input':kwargs['input_folder'],
                    'gold_folder':kwargs['gold_folder'],
                    'output_folder':kwargs['folder_output'],
                    'add_inname':kwargs['add_in_namefile'],
                    'list_landmark':list_lmrk_str,
                }
        print('parameter',parameter)

        OrientProcess = slicer.modules.aso_cbct
        process = slicer.cli.run(OrientProcess, None, parameter)

        return process

    def DownloadRef(self):
        webbrowser.open('https://google.com')

    def DicLandmark(self):
        dic = {'Head':
                {'Cranial Base' : ['Ba', 'S', 'N', 'RPo', 'LPo', 'RFZyg', 'LFZyg', 'C2', 'C3', 'C4'],

                'Upper' : ['RInfOr', 'LInfOr', 'LMZyg', 'RPF', 'LPF', 'PNS', 'ANS', 'A', 'UR3O', 'UR1O', 'UL3O', 'UR6DB', 'UR6MB', 'UL6MB', 'UL6DB', 'IF', 'ROr', 'LOr', 'RMZyg', 'RNC', 'LNC', 'UR7O', 'UR5O', 'UR4O', 'UR2O', 'UL1O', 'UL2O', 'UL4O', 'UL5O', 'UL7O', 'UL7R', 'UL5R', 'UL4R', 'UL2R', 'UL1R', 'UR2R', 'UR4R', 'UR5R', 'UR7R', 'UR6MP', 'UL6MP', 'UL6R', 'UR6R', 'UR6O', 'UL6O', 'UL3R', 'UR3R', 'UR1R'],

                'Lower' : ['RCo', 'RGo', 'Me', 'Gn', 'Pog', 'PogL', 'B', 'LGo', 'LCo', 'LR1O', 'LL6MB', 'LL6DB', 'LR6MB', 'LR6DB', 'LAF', 'LAE', 'RAF', 'RAE', 'LMCo', 'LLCo', 'RMCo', 'RLCo', 'RMeF', 'LMeF', 'RSig', 'RPRa', 'RARa', 'LSig', 'LARa', 'LPRa', 'LR7R', 'LR5R', 'LR4R', 'LR3R', 'LL3R', 'LL4R', 'LL5R', 'LL7R', 'LL7O', 'LL5O', 'LL4O', 'LL3O', 'LL2O', 'LL1O', 'LR2O', 'LR3O', 'LR4O', 'LR5O', 'LR7O', 'LL6R', 'LR6R', 'LL6O', 'LR6O', 'LR1R', 'LL1R', 'LL2R', 'LR2R'],

                }}

        return dic

    def existsLandmark(self,folderpath,reference_folder):
        out = None
        if folderpath != '' and reference_folder != '':
            input_lm = []
            input_json = super().search(folderpath,'json')['json']
            for file in input_json:
                for lm in self.ListLandmarksJson(file):
                    if lm not in input_lm:
                        input_lm.append(lm)
            
            gold_json = super().search(reference_folder,'json')['json']
            gold_lm = self.ListLandmarksJson(gold_json[0])

            available_lm = [lm for lm in input_lm if lm in gold_lm]
            
            dic = self.DicLandmark()['Head']
            list_lm = []
            for key in dic.keys():
                list_lm.extend(dic[key])
            
            not_available_lm = [lm for lm in list_lm if lm not in available_lm]
            
            out = {key:False for key in not_available_lm}

        return out

    def ListLandmarksJson(self,json_file):
        
        with open(json_file) as f:
            data = json.load(f)
        
        return [data["markups"][0]["controlPoints"][i]['label'] for i in range(len(data["markups"][0]["controlPoints"]))]
        
    def Sugest(self):
        return ['Ba','S','N','RPo','LPo','ROr','LOr']


    def __CheckboxisChecked(self,diccheckbox : dict):
        out=''
        listchecked = []
        if not len(diccheckbox) == 0:
            for checkboxs in diccheckbox.values():
                for checkbox in checkboxs:
                    if checkbox.isChecked():
                        listchecked.append(checkbox.text)
        return listchecked