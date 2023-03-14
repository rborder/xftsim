import warnings
import numpy as np
import pandas as pd
import math
import functools

from typing import Iterable, Union
from nptyping import NDArray, Shape, Float, Int, Object 
from collections.abc import Sequence

import xftsim as xft


import traceback
import warnings
import sys

# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))

# warnings.showwarning = warn_with_traceback

from xftsim.utils import ensure2D, cartesian_product, ids_from_n_generation, paste, merge_duplicates
## superclass not to be used
class XftIndex:
    def validate(self):
        assert self._coord_variables is not None
        assert self._index_variables is not None
        assert self._dimension is not None

    @property
    def frame(self):
        # self._frame[self._dimension] = self.unique_identifier
        # self._frame.set_index(self._dimension, inplace=True)
        return(self._frame)
    @frame.setter
    def frame(self, dataframe):
        # dataframe[self._dimension] = xft.utils.unique_identifier(dataframe, self._index_variables)
        # dataframe.set_index(self._dimension, inplace=True)
        self._frame = dataframe
        new_index = pd.Series(data=self.unique_identifier,
                              name=self._dimension)
        self._frame = dataframe.set_index(new_index)
        # self._frame[self._dimension] = self.unique_identifier
        # self._frame.set_index(self._dimension, inplace=True)
        # return(self._frame)        

    def frame_copy(self):
        return(self.frame.copy())

    @property
    def unique_identifier(self):
        return xft.utils.unique_identifier(self.frame, self._index_variables)
        # return paste([self._frame[variable].values for variable in self._index_variables], sep=".")
    
    @property
    def coords(self):
        coords = {self._dimension:self.unique_identifier}
        coords.update({variable:self.frame[variable].values for variable in self._coord_variables})
        return coords

    @property
    def coord_dict(self):
        return {key:(self._dimension, value) for (key,value) in self.coords.items()}

    @property
    def coord_frame(self):
        frame = pd.DataFrame.from_dict(self.coords)
        frame.set_index(self._dimension, drop=False, inplace=True)
        return frame

    @property
    def coord_mindex(self):
        cm = pd.MultiIndex.from_frame(self.coord_frame)
        return cm

    @property
    def coord_index(self):
        return pd.Index(self.unique_identifier)

    def __getitem__(self, arg): ## TODO: optimize to avoid reconstruction
        frame = self.frame
        ## subsetting for dict
        if isinstance(arg, dict):
            subset = np.full(frame.shape[0], fill_value=True, dtype=bool)
            for (key,value) in arg.items():
                if not key in frame.columns:
                    raise KeyError
                elif pd.api.types.is_list_like(value):            
                    subset *= frame[key].isin(value).values
                else:                 
                    subset *= (frame[key].values == value)
            return self.__class__(frame=frame.loc[subset,:])
        else:
            return self.__class__(frame=frame.loc[arg])

    def iloc(self, key): ## TODO: optimize to avoid reconstruction
        return self.__class__(frame=self.frame.iloc[key])

    def merge(self, other):
        if self.__class__ != other.__class__: raise TypeError()
        if self._dimension != other._dimension: raise TypeError()
        frame = pd.concat([self.frame, other.frame])
        frame = frame[~frame.duplicated()]
        if hasattr(self, 'generation'):
            return self.__class__(frame=frame, generation=self.generation)
        else: 
            return self.__class__(frame=frame)

    @staticmethod
    def reduce_merge(args):
        return functools.reduce(XftIndex.merge,args)

    def stack(self, other):
        if self.__class__ != other.__class__: raise TypeError()
        if self._dimension != other._dimension: raise TypeError()
        frame = pd.concat([self.frame, other.frame])
        if hasattr(self, 'generation'):
            return self.__class__(frame=frame, generation=self.generation)
        else: 
            return self.__class__(frame=frame)

    ## downsample at random
    def at_most(self, n_new):
        if self.n > n_new:
            # frame = self.frame.iloc[np.sort(np.random.choice(self.n, n_new))]
            frame = self.frame.iloc[:n_new]
        else: 
            frame = self.frame
        if hasattr(self, 'generation'):
            return self.__class__(frame=frame, generation=self.generation)
        else: 
            return self.__class__(frame=frame)


class SampleIndex(XftIndex):
    def __init__(self,
                 iid: Iterable = None, ## individual id
                 fid: Iterable = None, ## family id
                 sex: Iterable = None, ## biological sex
                 frame: pd.DataFrame = None,
                 n: int = None,
                 generation: int = 0,
                 ):
        self.generation = generation
        self._dimension = "sample"
        self._index_variables = np.array(['iid', 'fid'])
        self._coord_variables = np.array(['iid', 'fid', 'sex'])

        ## either provide iid XOR n XOR frame
        assert (iid is not None) ^ (frame is not None) ^ (n is not None)

        ## if iid provided, populate fid/sex if needed, generate frame
        if iid is not None:
            iid = np.array(iid)
            n = iid.shape[0]
            if fid is None:
                fid = iid
            if sex is None:
                sex = np.tile([0,1], math.ceil(n/2))[:n] ## ensure proper length
            self.frame = pd.DataFrame.from_dict({'iid':iid, 'fid':fid, 'sex':sex})
        ## if n provided, populate iid/fid/sex, generate frame
        elif n is not None:
            iid = ids_from_n_generation(n, generation)
            fid = iid
            sex = np.tile([0,1], math.ceil(n/2))[:n]
            self.frame = pd.DataFrame.from_dict({'iid':iid, 'fid':fid, 'sex':sex})
        ## otherwise just use provided frame
        elif frame is not None:
            self.frame = frame


    ## access data frame columns as properities
    @property
    def iid(self):
        return self.frame['iid'].values
    # @iid.setter
    # def iid(self, value):
        # self._frame['iid']=value
    @property
    def fid(self):
        return self.frame['fid'].values
    # @fid.setter
    # def fid(self, value):
        # self._frame['fid']=value
    @property
    def sex(self):
        return self.frame['sex'].values
    # @sex.setter
    # def sex(self, value):
        # self._frame['sex']=value
        
    @property
    def n(self):
        return self.iid.shape[0]
    
    @property
    def n_fam(self):
        return np.unique(self.fid).shape[0]
    
    @property
    def n_female(self):
        return np.sum(self.sex==0).astype(int)
    
    @property
    def n_male(self):
        return np.sum(self.sex==1).astype(int)

    def __repr__(self, short=False):
        cf = self.frame
        mi_repr = cf.__repr__().split("\n")
        if len(mi_repr) > 10:
            mi_repr = mi_repr[:3] + ["            ..."] + mi_repr[-3:]
        output = [
                  "<SampleIndex>",
                  f"  {self.n} indviduals from {self.n_fam} families",
                  f"  {self.n_female} biological females",
                  f"  {self.n_male} biological males",
                  ] + mi_repr
        if short:
            return output[1:4]
        else:
            return "\n".join(output)

    def __eq__(self, other):
        return np.all(self.frame == other.frame)


class DiploidVariantIndex(XftIndex):
    def __init__(self,
                 vid: NDArray[Shape["*"], Object] = None, ## variant id
                 chrom: NDArray[Shape["*"], Int] = None, ## chromosome
                 zero_allele: NDArray[Shape["*"], Object] = None, ## allele coding
                 one_allele: NDArray[Shape["*"], Object] = None, 
                 af: Iterable = None,
                 annotation_array: Union[NDArray, pd.DataFrame] = None,
                 annotation_names: Iterable = None,
                 frame: pd.DataFrame = None,
                 m: int = None,
                 n_chrom: int = 1,
                 h_copy: NDArray[Shape["*"], Object] = None,
                 pos_bp: Iterable = None,
                 pos_cM: Iterable = None,                 
                 ):
        self._dimension = "variant"
        self._index_variables = ["vid", "hcopy"]
        self._coord_variables = ["vid", "chrom","hcopy","zero_allele",
                                 "one_allele", "af", "pos_bp", "pos_cM"]

        ## provide vid XOR m XOR frame
        assert (vid is not None) ^ (m is not None) ^ (frame is not None)
        if frame is None:
            ## set generic vids if not provided
            if m is not None:
                vid = np.arange(m).astype(str)
            else: 
                vid = np.array(vid)
                m = vid.shape[0]
                if self.ploidy =="Diploid":
                    assert np.unique(vid).shape[0] == vid.shape[0], "Duplicate variant IDs"
            ## set generic chromosomes if missing
            if chrom is None:
                chrom = np.repeat(np.arange(n_chrom), 
                                  math.ceil(m/n_chrom))[:m]
            ## set generic zero/one alleles if either is missing
            if zero_allele is None or one_allele is None:
                zero_allele = np.repeat(['A'], m)
                one_allele = np.repeat(['G'], m)
            ## set afs to NaN if missing
            if af is not None:
                assert len(af) == m, "number of variants and afs do not match"
            else:
                af = np.full(m, fill_value=np.NaN)
            ## populate positions if needed
            if pos_bp is not None:
                assert len(pos_bp) == m, "number of variants and pos_bp do not match"
            else:
                pos_bp = np.full(m, fill_value=np.NaN)
            if pos_cM is not None:
                assert len(pos_cM) == m, "number of variants and pos_cM do not match"
            else:
                pos_cM = np.full(m, fill_value=np.NaN)
            ## populate hcopy if needed
            if h_copy is None:
                h_copy = np.repeat("d",m)
            self.frame = pd.DataFrame.from_dict({
                       "vid":vid, 
                       "chrom":chrom,
                       "zero_allele":zero_allele, 
                       "one_allele":one_allele,
                       "af":af,
                       "hcopy":h_copy,
                       "pos_bp":pos_bp,
                       "pos_cM":pos_cM,
                       })
        ## otherwise use provided data frame
        else:
            self.frame = frame
            ## TODO verify df is valid
            ## populate possibly missing elements
            if not "hcopy" in frame.columns.values:
                self.frame["hcopy"]=np.repeat("d",m)
            if not "pos_bp" in frame.columns.values:
                self.frame["pos_bp"]=np.repeat(np.NaN,m)
            if not "pos_cM" in frame.columns.values:
                self.frame["pos_cM"]=np.repeat(np.NaN,m)
        self._annotation_array = annotation_array
        self._annotation_names = annotation_names
        if annotation_array is not None:
            if annotation_names is None:
                self._annotation_names = np.char.add('a_',np.arange(annotation_array.shape[1]).astype(str))
            self._annotation = pd.DataFrame.from_dict({key:value for (key,value) in zip(self._annotation_names, annotation_array.T)})
            self.frame = pd.concat([self.frame, self._annotation],1)
            self._coord_variables = np.concatenate(self._coord_variables,
                                                   self._annotation_names)
        else:
            self._annotation = None


    @property
    def vid(self):
        return self.frame['vid'].values
    # @vid.setter
    # def vid(self, value):
        # self._frame['vid'] = value
    @property
    def chrom(self):
        return self.frame['chrom'].values
    # @chrom.setter
    # def chrom(self, value):
        # self._frame['chrom'] = value
    @property
    def one_allele(self):
        return self.frame['one_allele'].values
    # @one_allele.setter
    # def one_allele(self, value):
        # self._frame['one_allele'] = value
    @property
    def zero_allele(self):
        return self.frame['zero_allele'].values
    # @zero_allele.setter
    # def zero_allele(self, value):
        # self._frame['zero_allele'] = value
    @property
    def hcopy(self):
        return self.frame['hcopy'].values
    # @hcopy.setter
    # def hcopy(self, value):
        # self._frame['hcopy'] = value
    @property
    def af(self):
        return self.frame['af'].values
    # @af.setter
    # def af(self, value):
        # self._frame['af'] = value
    
    @property
    def pos_bp(self):
        return self.frame['pos_bp'].values
    
    @property
    def pos_cM(self):
        return self.frame['pos_cM'].values
    
    @property
    def ploidy(self):
        return "Diploid"

    @property
    def annotation(self):
        return self._annotation
    
    @property
    def annotation_array(self):
        return self._annotation_array

    @property
    def annotation_names(self):
        return self._annotation_names

    @property
    def m(self):
        return np.unique(self.vid).shape[0]

    @property
    def n_chrom(self):
        return np.unique(self.chrom).shape[0]

    @property
    def n_annotations(self):
        if self.annotation is not None:
            return self.annotation.shape[1]
        else:
            return 0

    @property
    def maf(self):
        if not np.any(self.af==np.NaN):
            return np.full(self.m, fill_value=np.NaN)
        else:
            tmp = 1 - self.af
            return np.where(tmp <= self.af, tmp, self.af)
    

    def __repr__(self):
        cf = self.frame
        mi_repr = cf.__repr__().split("\n")
        if np.any(self.af == np.NaN) is None:
            af_str = "  allele frequencies unknown"
        else:
            af_str = f"  MAF ranges from {min(self.maf)} to {max(self.maf)}"
        output = [
               "<"+self.ploidy+"VariantIndex>",
               f"  {self.m} diploid variants on {self.n_chrom} chromosome(s)",
               af_str,
               f"  {self.n_annotations} annotation(s) ",
               ] + mi_repr
        return "\n".join(output)

    def __eq__(self, other):
        return np.all(self.coord_frame == other.coord_frame)

    def to_haploid(self):
        hframe = pd.DataFrame.from_dict({
           "vid":np.repeat(self.vid, 2), 
           "chrom":np.repeat(self.chrom, 2),
           "zero_allele":np.repeat(self.zero_allele, 2), 
           "one_allele":np.repeat(self.one_allele, 2),
           "af":np.repeat(self.af, 2),
           "hcopy":np.tile(['0','1'],self.vid.shape[0]),
           "pos_bp":np.repeat(self.pos_bp, 2),
           "pos_cM":np.repeat(self.pos_cM, 2),
           })
        if self._annotation_array is not None:
            self._annotation_array = np.repeat(self._annotation_array,2,axis=1)
            self._annotation_names = np.repeat(self._annotation_array,2)

        return HaploidVariantIndex(frame=hframe,
                            annotation_array=self.annotation_array,
                            annotation_names=self.annotation_names)

    def annotate(self):
        raise NotImplementedError ## TODO



class HaploidVariantIndex(DiploidVariantIndex):
    @property
    def ploidy(self):
        return "Haploid"

    def to_diploid(self):
        dframe = pd.DataFrame.from_dict({
           "vid":self.vid[::2],
           "chrom":self.chrom[::2],
           "zero_allele":self.zero_allele[::2],
           "one_allele":self.one_allele[::2],
           "af":self.af[::2],
           "hcopy":np.repeat(['d'],self.vid.shape[0]//2),
           "pos_bp":self.pos_bp[::2],
           "pos_cM":self.pos_cM[::2],
           })
        if self._annotation_array is not None:
            self._annotation_array = self._annotation_array[:,::2]
            self._annotation_names = self._annotation_array[::2]

        return DiploidVariantIndex(frame=dframe,
                            annotation_array=self.annotation_array,
                            annotation_names=self.annotation_names)


class ComponentIndex(XftIndex):
    def __init__(self,
                 phenotype_name: Iterable = None, # names of phenotypes
                 component_name: Iterable = None, # names of phenotype components
                 vorigin_relative: Iterable = None, # relative of phenotype origin
                 frame: pd.DataFrame = None,
                 k_total: int = None,
                 ):
        self._dimension = "component"
        self._index_variables = np.array(['phenotype_name', 'component_name', 'vorigin_relative'])
        self._coord_variables = np.array(['phenotype_name', 'component_name', 'vorigin_relative'])

        ## provide phenotype_name XOR frame XOR k_total
        assert (phenotype_name is not None) ^ (k_total is not None)^ (frame is not None), "provide phenotype_name OR k_total OR frame"

        ## set generic phenotype names if not provided
        if frame is None:
            if phenotype_name is None:
                assert component_name is None and vorigin_relative is None, "provide phenotype_name with component_name / vorigin_relative"
                phenotype_name = np.arange(k_total).astype(str)
            else:
                phenotype_name = np.array(phenotype_name).astype(str)
                k_total = phenotype_name.shape[0]
            if component_name is None:
                component_name = np.repeat("generic", k_total)
            else:
                component_name = np.array(component_name).astype(str)
            if vorigin_relative is None:
                vorigin_relative = np.repeat(-1, k_total).astype(int)
            else:
                vorigin_relative = np.array(vorigin_relative)
            ## check for duplicates:
            # phenotype_name,component_name,vorigin_relative = merge_duplicates([phenotype_name,
            #                                                                    component_name,
            #                                                                    vorigin_relative])
            self.frame = pd.DataFrame.from_dict({'phenotype_name':phenotype_name,
                                                 'component_name':component_name,
                                                 'vorigin_relative':vorigin_relative.astype(int)})
        elif frame is not None:
            self.frame = frame

    ## function to return component index with vorigin_relative = -1
    ## only intended for use when all vorigin_relative == x != -1
    def to_vorigin(self, origin):
        frame = self.frame_copy()
        frame.vorigin_relative = origin
        return ComponentIndex.from_frame(frame)

    def to_proband(self):
        if len(np.unique(self.vorigin_relative)) > 1 or np.any(self.vorigin_relative==-1):
            raise RuntimeError
        else:
            return self.to_vorigin(-1)
            # frame = self.frame_copy()
            # frame.vorigin_relative = -1
            # return ComponentIndex.from_frame(frame)
            
    @property
    def phenotype_name(self):
        return self.frame['phenotype_name']
    @phenotype_name.setter
    def phenotype_name(self, value):
        self._frame['phenotype_name'] = value

    @property
    def unique_identifier(self):
        fr = self.frame[['phenotype_name','component_name','vorigin_relative']]
        fr['vorigin_relative'].replace([-1,0,1],['proband','mother','father'], inplace=True)
        return xft.utils.unique_identifier(fr, ['phenotype_name','component_name','vorigin_relative'])

    @property
    def component_name(self):
        return self.frame['component_name']
    @component_name.setter
    def component_name(self, value):
        self._frame['component_name'] = value

    @property
    def vorigin_relative(self):
        return self.frame['vorigin_relative']
    @vorigin_relative.setter
    def vorigin_relative(self, value):
        self._frame['vorigin_relative'] = value

    @property
    def k_total(self):
        return self.phenotype_name.shape[0]

    @property
    def k_phenotypes(self):
        return np.unique(self.phenotype_name).shape[0]

    @property
    def k_components(self):
        return np.unique(self.component_name).shape[0]

    @property
    def k_relative(self):
        return np.unique(self.vorigin_relative).shape[0]


    @property
    def depth(self): ## generational depth from binary relative encoding
        if len(self.vorigin_relative) > 0:
            return math.floor(math.log2(np.max(self.vorigin_relative)+2)) + 1
        else:
            return np.NaN 

    def __repr__(self):
        mi_repr = self.frame.__repr__().split("\n")
        if len(mi_repr) > 10:
            mi_repr = mi_repr[:4] + ["            ..."] + mi_repr[-2:]
        if self.depth == 1:
            depth_str = f"{self.depth} generation"
        else:
            depth_str = f"{self.depth} generations"
        if self.k_components == 1:
            kcomp_str = f"{self.k_components} component"
        else:
            kcomp_str = f"{self.k_components} components"
        if self.k_phenotypes == 1:
            kphen_str = f"{self.k_phenotypes} phenotype"
        else:
            kphen_str = f"{self.k_phenotypes} phenotypes"      
        return "\n".join([
                         "<ComponentIndex>",
                         f"  {kcomp_str} of {kphen_str} spanning {depth_str}",
                         ] + mi_repr)

    def __eq__(self, other):
        return np.all(self.coord_frame == other.coord_frame)

    @staticmethod
    def from_frame(df: pd.DataFrame):
        phenotype_name = df.phenotype_name.values
        component_name = df.component_name.values
        vorigin_relative = df.vorigin_relative.values
        return ComponentIndex(phenotype_name, component_name, vorigin_relative)

    @staticmethod
    def from_arrays(phenotype_name: Iterable,
                                   component_name: Iterable,
                                   vorigin_relative: Iterable,
                                   ):
        return ComponentIndex(phenotype_name, component_name, vorigin_relative)

    @staticmethod
    def from_product(phenotype_name: Iterable,
                                    component_name: Iterable,
                                    vorigin_relative: Iterable,
                                   ):
        phenotype_name, component_name, vorigin_relative = cartesian_product(phenotype_name, 
                                                                               component_name, 
                                                                               vorigin_relative)
        return ComponentIndex(phenotype_name, component_name, vorigin_relative)

## TODO
def sampleIndex_from_plink():
    raise NotImplementedError

## TODO
def sampleIndex_from_VCF():
    raise NotImplementedError

## TODO
def variantIndex_from_plink():
    raise NotImplementedError

## TODO
def variantIndex_from_VCF():
    raise NotImplementedError

## tests
def _test_SampleIndex():
    generation = 0
    n=10
    iid = np.char.add(str(generation)+"_",np.arange(n).astype(str))
    fid = iid
    sex = np.tile(np.arange(2),math.ceil(n/2))[:n]
    ## should work
    default = SampleIndex(n=n)
    assert default == SampleIndex(n=n, generation = generation)
    assert default == SampleIndex(iid=iid)
    assert default == SampleIndex(iid=iid, generation = generation)
    assert default == SampleIndex(iid=iid, fid=fid)
    assert default == SampleIndex(iid=iid, fid=fid, generation = generation)
    assert default == SampleIndex(iid=iid, fid=fid, sex=sex)
    assert default == SampleIndex(iid=iid, fid=fid, sex=sex, generation = generation)
    return 0 ## TODO error cases

def _test_VariantIndex():
    pass

def _test_DiploidVariantIndex():
    pass

def _test_HaploidVariantIndex():
    pass

def _test_ComponentIndex():
    pass

def _test_ComponentIndex_from_arrays():
    pass

def _test_ComponentIndex_from_product():
    pass

def _test_sampleIndex_from_plink():
    pass

def _test_sampleIndex_from_VCF():
    pass

def _test_variantIndex_from_plink():
    pass

def _test_variantIndex_from_VCF():
    pass
