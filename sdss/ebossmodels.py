#!/usr/bin/env python

from simqso import sqmodels
from simqso import sqgrids as grids

cont_models = {
  'bossdr9':[[(-1.50,0.3),(-0.50,0.3),(-0.37,0.3),(-1.70,0.3),(-1.03,0.3)],
             [1100.,5700.,9730.,22300.]],
  'dr9expdust':[[(-0.50,0.3),(-0.30,0.3),(-0.37,0.3),
                 (-1.70,0.3),(-1.03,0.3)],
                 [1100.,5700.,9730.,22300.]],
  'ebossdr14':[[(-1.5,0.3),(-0.4,0.3)],[1200.]],
  'dr14expdust':[[(-1.2,0.3),(-0.3,0.3)],[1200.]],
}

emline_models = {
  'ebossdr14':{
    'scaleEWs':{'LyAb':1.1,'LyAn':1.1,'CIVb':0.75,'CIVn':0.75,
                 'CIII]b':0.8,'CIII]n':0.8,'MgIIb':0.8,'MgIIn':0.8,
                 'Hbeta':1.2,},#'HAb':1.0,'HAn':1.0},
    'scaleLogScatter':{'HAb':5,},#'HAn':5},
    },
}

dustem_models = {
  #
  'LR17':{'sublimdust':[(0.05,None),(1800.,None)],
             'hotdust':[(0.2,None),(880.,None)]},
  #
  'LR17b':{'sublimdust':[(0.05,None),(1800.,None)],
             'hotdust':[(0.08,None),(880.,None)]},
  #
  'GHW06':{'hotdust':[(0.1,None),(1260.,None)]},
  #
  'GHW06b':{'sublimdust':[(0.03,None),(1800.,None)],
           'hotdust':[(0.07,None),(1260.,None)]},
  #
  'dr14expdust':{'sublimdust':[(0.03,None),(1800.,None)],
                 'hotdust':[(0.06,None),(880.,None)]},
}

qso_models = {
  #
  'bossdr9':{'continuum':'bossdr9','emlines':'bossdr9','iron':'bossdr9'},
  #
  'dr9expdust':{'continuum':'dr9expdust','emlines':'bossdr9',
                'iron':'bossdr9','dustext':'dr9expdust'},
  #
  'ebossdr14':{'continuum':'ebossdr14','emlines':'ebossdr14',
               'dustem':'LR17b','iron':'bossdr9'},
  #
  'ebossdr14ghwdust':{'continuum':'ebossdr14','emlines':'ebossdr14',
                      'dustem':'GHW06b','iron':'bossdr9'},
  #
  'dr14expdust':{'continuum':'dr14expdust','emlines':'ebossdr14',
                 'dustem':'dr14expdust','iron':'bossdr9',
                 'dustext':'dr9expdust'},
}

def add_continuum(qsos,model='ebossdr14',const=False):
	print "CONTINUUM: {}".format(model)
	try:
		slopes,breakpts = cont_models[model]
	except KeyError:
		if isinstance(model,basestring):
			slopes,breakpts = eval(model)
		else:
			slopes,breakpts = model
	if const:
		slopes = [ grids.ConstSampler(s[0]) for s in slopes]
	else:
		slopes = [ grids.GaussianSampler(*s) for s in slopes]
	print "CONTINUUM: {}".format(str(slopes))
	contVar = grids.BrokenPowerLawContinuumVar(slopes,breakpts)
	qsos.addVar(contVar)
	return qsos

def add_dust_emission(qsos,model='LR17',const=False):
	contVar = qsos.getVars(grids.ContinuumVar)[0]
	try:
		model = dustem_models[model]
		print "DUST EMISSION: {}".format(model)
	except KeyError:
		if isinstance(model,basestring):
			model = eval(model)
		else:
			pass
		print "DUST EMISSION: {}".format(str(model))
	dustVars = []
	for name,par in model.items():
		dustVar = grids.DustBlackbodyVar([grids.ConstSampler(par[0][0]),
		                                  grids.ConstSampler(par[1][0])],
	                                     name=name)
		dustVar.set_associated_var(contVar)
		dustVars.append(dustVar)
	qsos.addVars(dustVars)
	return qsos

def add_emission_lines(qsos,model='bossdr9',const=False):
	print "EMISSION LINES: {}".format(model)
	if model == 'bossdr9':
		emLineVar = sqmodels.BossDr9_EmLineTemplate(qsos.absMag,
		                                            NoScatter=const)
	elif model == 'yang16':
		emLineVar = sqmodels.get_Yang16_EmLineTemplate(qsos.absMag,
		                                               NoScatter=const)
	else:
		try:
			kwargs = emline_models[model]
		except KeyError:
			if isinstance(model,basestring):
				kwargs = eval(model)
			else:
				kwargs = model
		kwargs['NoScatter'] = const
		print "EMISSION LINES: {}".format(str(kwargs))
		emLineVar = grids.generateBEffEmissionLines(qsos.absMag,**kwargs)
	qsos.addVar(emLineVar)
	return qsos

def add_iron(qsos,wave,model='bossdr9',const=False):
	print "IRON TEMPLATE: {}".format(model)
	fetempl = grids.VW01FeTemplateGrid(qsos.z,wave,
	                                   scales=sqmodels.BossDr9_FeScalings)
	feVar = grids.FeTemplateVar(fetempl)
	qsos.addVar(feVar)
	return qsos

def add_dust_extinction(qsos,model='dr9expdust',const=False):
	if model == 'dr9expdust':
		dustscl = 0.033
		print "DUST EXTINCTION: {} with scale {}".format(model,dustscl)
		if const:
			s = grids.ConstSampler(dustscl)
		else:
			s = grids.ExponentialSampler(dustscl)
	else:
		raise ValueError
	dustVar = grids.SMCDustVar(s)
	qsos.addVar(dustVar)
	return qsos

