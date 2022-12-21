from dataclasses import dataclass,field ,astuple, asdict
from typing import Tuple, Union, List
import os
import glob


@dataclass(init=True)
class Upper:
    name1 : str = 'Upper'
    name2 : str = '_U_'

    def __str__(self) -> str:
        return 'Upper'



@dataclass(init=True)
class Lower :
    name1 : str = 'Lower'
    name2: str = '_L_'


    def __str__(self) -> str:
        return 'Lower'


@dataclass(init=True,repr=True)
class Jaw:
    upper : Upper = field(init = False, repr=False,default=Upper())
    lower : Lower = field(init=False, repr=False, default=Lower())
    actual : Union[Upper, Lower]

    def __init__(self,actual) -> None:
        self.actual = actual


    def inv(self):
        out =  self.upper
        if isinstance(self.actual,Upper):
            out = self.lower
        
    
        return str(out)



    def __str__(self) -> str:
        return str(self.actual)

    def __eq__(self,other):
        out = False
        if isinstance(other.actual,self.actual):
            out = True
        return out

    def __call__(self):

        return str(self.actual)







@dataclass(init=True,repr=True,eq=True,frozen=True)
class Jaw_File :
    json : str
    vtk : str
    jaw : Jaw
    name : str



@dataclass(init=True,repr=True,eq=True)
class Mouth_File:
    Upper : Union[Jaw_File , str]
    Lower : Union [Jaw_File, str]
    name : str




@dataclass(init=True,repr=True,eq=True,frozen=False)
class Files:
    list_file : List[Mouth_File] = field(init=False)
    folder : str 


    def __type_jaw__(self,name_file : str):
        out = None

        if True in [upper in name_file for upper in astuple(Upper())]:
            out =Upper()

        elif True in [upper in name_file for upper in astuple(Lower())]:
            out = Lower()

        if out is None:
            raise ValueError(f"dont found the jaw's type to {name_file}")
        return out


    def __name_file__(self,name_file : str):
        name_file = os.path.basename(name_file)
        name_file, _ = os.path.splitext(name_file)
        jaw = self.__type_jaw__(name_file)
        name_file = self.__remove_jaw__(name_file,jaw)

        
        return jaw, name_file


    def __remove_jaw__(self,name_file : str,jaw : Union[Upper,Lower]):
        work = False
        for st in astuple(jaw):
            if st.lower() in name_file.lower():
                index = name_file.lower().find(st.lower())
                name_file = name_file[:index]+name_file[index+len(st):]
                work = True
        
        if work :
            self.__remove_jaw__(name_file,jaw)

        return name_file

        




    def __len__(self):
        return len(self.list_file)

    def __iter__(self):
        self.iter=-1
        return self


    def __next__(self):
        self.iter += 1 
        if self.iter>= len(self.list_file):
            raise StopIteration

        
        
        return asdict(self.list_file[self.iter])



    def search(self,path,*args):
        """
        Return a dictionary with args element as key and a list of file in path directory finishing by args extension for each key

        Example:
        args = ('json',['.nii.gz','.nrrd'])
        return:
            {
                'json' : ['path/a.json', 'path/b.json','path/c.json'],
                '.nii.gz' : ['path/a.nii.gz', 'path/b.nii.gz']
                '.nrrd.gz' : ['path/c.nrrd']
            }
        """
        arguments=[]
        for arg in args:
            if type(arg) == list:
                arguments.extend(arg)
            else:
                arguments.append(arg)
        out = {key: [i for i in glob.iglob(os.path.normpath("/".join([path,'**','*'])),recursive=True) if i.endswith(key)] for key in arguments}

        for key , values in out.items():
            lst = []
            for value in values :
                if os.path.isfile(value):
                    lst.append(value)
            out[key]=lst

        return out 


@dataclass
class Files_vtk(Files):
    def __post_init__(self):
        self.list_file= []
        dic =self.search(self.folder,'vtk')
        list_vtk = dic['vtk']

        dic ={}
        for vtk in list_vtk:
            jaw , name = self.__name_file__(vtk)
            if name in dic:
                dic[name].append(vtk)
            else :
                dic[name]= [vtk]


        for key , value in dic.items():
            if len(value)==2:
                vtk1 = value[0]
                vtk2 = value[1]
                jaw1, name1 = self.__name_file__(vtk1)
                if isinstance(jaw1,Lower):
                    vtk1, vtk2 = vtk2, vtk1

                self.list_file.append(Mouth_File(vtk1,vtk2,name1))




@dataclass
class Files_vtk_json(Files):
    def __post_init__(self):
        self.list_file= []
        dic =self.search(self.folder,'vtk','json')
        list_json = dic['json']
        list_vtk = dic['vtk']
        # for file in database:
        #     if os.path.isfile(file):
        #         _ , extension = os.path.splitext(file)
        #         jaw, name_file = self.__name_file__(file)
        #         fil = File(file,name_file,jaw,extension)
        #         if extension == '.vtk':
        #             list_vtk.append(fil)

        #         elif extension == '.json':
        #             list_json.append(fil)

        list_upper = []
        list_lower = []
        for vtk in list_vtk:
            vtk_jaw , vtk_name = self.__name_file__(vtk)
            for json in list_json:
                json_jaw , json_name = self.__name_file__(json)
                # print('name',vtk_name,json_name)
                if vtk_name in json_name and vtk_jaw==json_jaw :
                    fil = Jaw_File(json, vtk, json_jaw,vtk_name)
                    if isinstance(fil.jaw,Upper):
                        list_upper.append(fil)
                    else:
                        list_lower.append(fil)
                    
                    list_json.remove(json)



        print('len upper',len(list_upper))
        print('lne lower',len(list_lower))
        for upper in list_upper:
            for lower in list_lower:
                if upper.name == lower.name :
                    fil = Mouth_File(upper,lower,upper.name)
                    self.list_file.append(fil)
                    list_lower.remove(lower)


