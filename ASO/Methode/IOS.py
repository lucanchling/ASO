from Methode.Methode import Methode
import slicer
import webbrowser
import glob
import os

class Auto_IOS(Methode):
    def __init__(self, widget):
        super().__init__(widget)


    def NumberScan(self, scan_folder: str):
            
        return len(super().search(scan_folder,'vtk')['vtk'])


    def TestScan(self, scan_folder: str):
        out = None
        if self.NumberScan(scan_folder) == 0 :
            out = 'Give folder with vkt file'
        return out

    def TestModel(self, model_folder: str) -> str:
        return super().TestModel(model_folder)
        
    def TestReference(self, ref_folder: str):
        list = glob.glob(ref_folder+'/*vtk')
        out = None
        if len(list) == 0:
            out = 'Please choose a folder with json file'
        elif len(list)>2:
            out = 'Too many json file '
        return out

    def TestCheckbox(self,dic_checkbox) -> str:
        list_teeth, list_landmark, mix, jaw = self.__CheckboxisChecked(dic_checkbox)
        out = None
        if len(list_teeth)< 3:
             out = 'Give minimum 3 teeth,'
        if len(list_landmark)==0:
            out += "Give minimum 1 landmark,"

        if len(jaw)<1 :
            out+= 'Choose one jaw,'
        return out


    def DownloadRef(self):
        webbrowser.open('https://github.com/HUTIN1/ASO/releases/tag/v1.0.0')

    def DownloadModels(self):
        return super().DownloadModels()


    def __Model(self,path):
        model = self.search(path,'pth')['pth'][0]

        return model


        

    def TestProcess(self,**kwargs) -> str:
        out  = ''

        scan = self.TestScan(kwargs['input_folder'])
        if isinstance(scan,str):
            out = out + f'{scan},'

        reference =self.TestReference(kwargs['gold_folder'])
        if isinstance(reference,str):
            out = out + f'{reference},'

        if kwargs['folder_output'] == '':
            out = out + "Give output folder,"

        testcheckbox = self.TestCheckbox(kwargs['dic_checkbox'])
        if isinstance(testcheckbox,str):
            out = out + f"{testcheckbox},"

        if kwargs['add_in_namefile']== '':
            out = out + "Give something to add in name of file,"


        if out != '':
            out=out[:-1]

        else : 
            out = None

        return out


    def Process(self, **kwargs):
        list_teeth, list_landmark , mix, jaw = self.__CheckboxisChecked(kwargs['dic_checkbox'])
        print('output checkbox is checked',list_teeth, list_landmark , mix)

        path_seg = os.path.join(slicer.util.tempDirectory(), 'seg')
        path_preor = os.path.join(slicer.util.tempDirectory(), 'PreOr')

        if not os.path.exists(path_seg):
            os.mkdir(os.path.join(path_seg))


        if not os.path.exists(path_preor):
            os.mkdir(path_preor)


        parameter_seg = {'input':kwargs['input_folder'],
                        'output':path_seg,
                        'rotation':40,
                        'resolution':320,
                        'model':self.__Model(kwargs['model_folder_segor']),
                        'predictedId':'Universal_ID',
                        'sepOutputs':False,
                        'chooseFDI':0,
                        'logPath':kwargs['logPath']
                        }

        parameter_pre_aso= {'input':path_seg,
                        'gold_folder':kwargs['gold_folder'],
                        'output_folder':path_preor,
                        'add_inname':kwargs['add_in_namefile'],
                        'list_teeth':','.join(list_teeth) }

        parameter_aliios ={'input':path_preor,
                            'dir_models':kwargs['model_folder_ali'],
                            'landmarks':' '.join(list_landmark),
                            'teeth':' '.join(list_teeth),
                            'save_in_folder':'false',
                            'output_dir':path_preor
                            }

        parameter_semi_aso= {'input':path_preor,
                            'gold_folder':kwargs['gold_folder'],
                            'output_folder':kwargs['folder_output'],
                            'add_inname':kwargs['add_in_namefile'],
                            'list_landmark':','.join(mix),
                            'jaw':'/'.join(jaw)}

        print('parameter pre aso',parameter_pre_aso)
        print('parameter seg',parameter_seg)
        print('parameter aliios ',parameter_aliios)
        print('parameter semi ios',parameter_semi_aso)

        PreOrientProcess = slicer.modules.pre_aso_ios
        SegProcess = slicer.modules.crownsegmentationcli
        aliiosProcess = slicer.modules.ali_ios
        OrientProcess = slicer.modules.semi_aso_ios

# {'Process':SegProcess,'Parameter':parameter_seg},{'Process':PreOrientProcess,'Parameter':parameter_pre_aso},
        list_process = [{'Process':SegProcess,'Parameter':parameter_seg},
        {'Process':PreOrientProcess,'Parameter':parameter_pre_aso},
        {"Process":aliiosProcess,"Parameter":parameter_aliios},
        {'Process':OrientProcess,'Parameter':parameter_semi_aso}]
#
        return list_process

    def DicLandmark(self):
        dic = {'Landmark':{'Occlusal':['DB','MB','O'],'Cervical':['CB','CL','OIP','R','RIP']}}

        return dic





    def existsLandmark(self,folderpath,reference_folder,model_folder):

        return None


    def Sugest(self):
        return ['UR6','UL1','UL6','LL6','LR1','LR6','O']


    def __CheckboxisChecked(self,diccheckbox : dict):
        dic = {'UR8': 1, 'UR7': 2, 'UR6': 3, 'UR5': 4, 'UR4': 5, 'UR3': 6, 'UR2': 7, 'UR1': 8, 'UL1': 9, 'UL2': 10, 'UL3': 11, 
       'UL4': 12, 'UL5': 13, 'UL6': 14, 'UL7': 15, 'UL8': 16, 'LL8': 17, 'LL7': 18, 'LL6': 19, 'LL5': 20, 'LL4': 21, 'LL3': 22, 
       'LL2': 23, 'LL1': 24, 'LR1': 25, 'LR2': 26, 'LR3': 27, 'LR4': 28, 'LR5': 29, 'LR6': 30, 'LR7': 31, 'LR8': 32}

        teeth = []
        landmarks= []
        jaw = []
        mix = []
        if not len(diccheckbox) == 0:

            for checkboxs in diccheckbox['Teeth']['Adult'].values():
                for checkbox in checkboxs:
                    if checkbox.isChecked():
                        teeth.append(checkbox.text)


            for checkboxs in diccheckbox['Landmark'].values():
                for checkbox in checkboxs:
                    if checkbox.isChecked():
                        landmarks.append(checkbox.text)


            for tooth in teeth :
                for landmark in landmarks :
                    mix.append(f'{tooth}{landmark}')


            for key , checkbox in diccheckbox['Jaw'].items():
                if checkbox.isChecked():
                    jaw.append(key)

        return teeth , landmarks, mix, jaw














class Semi_IOS(Auto_IOS):

       


    def TestScan(self, scan_folder: str):
        out = None
        dic = self.search(scan_folder,'vtk','json')
        if len(dic['vtk']) != len(dic['json']): 
            print('dif ',len(dic['vtk']) - len(dic['json']))
            # out = 'Give folder with the same number of vkt file and json file'
        return out 


    def __CheckboxisChecked(self,diccheckbox : dict):
        
        teeth = []
        landmarks= []
        jaw = []
        mix = []
        print(diccheckbox)
        if not len(diccheckbox) == 0:

            for checkboxs in diccheckbox['Teeth']['Adult'].values():
                for checkbox in checkboxs:
                    if checkbox.isChecked():
                        teeth.append(checkbox.text)


            for checkboxs in diccheckbox['Landmark'].values():
                for checkbox in checkboxs:
                    if checkbox.isChecked():
                        landmarks.append(checkbox.text)


            for tooth in teeth :
                for landmark in landmarks :
                    mix.append(f'{tooth}{landmark}')


            for key , checkbox in diccheckbox['Jaw'].items():
                if checkbox.isChecked():
                    jaw.append(key)

        return teeth , landmarks, mix, jaw



    def Process(self, **kwargs):
        teeth, landmark , mix , jaw = self.__CheckboxisChecked(kwargs['dic_checkbox'])

        parameter= {'input':kwargs['input_folder'],'gold_folder':kwargs['gold_folder'],'output_folder':kwargs['folder_output'],'add_inname':kwargs['add_in_namefile'],'list_landmark':','.join(mix),'Jaw':'/'.join(jaw)}


        print('parameter',parameter)
        OrientProcess = slicer.modules.semi_aso_ios

        return [{'Process':OrientProcess,'Parameter':parameter}]


    def Sugest(self):
        out = ['O','UL6','UL1','UR1','UR6','LL6','LL1','LR1','LR6']
        return out