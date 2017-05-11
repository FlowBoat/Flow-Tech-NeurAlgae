# FlowTech | NeurAlgae
## 2017 CWSF Science Fair | NeurAlgae: HAB Prediction Using Machine Learning Algorithms

Describes and trains a neural network for the analysis and prediction of algal bloom data
Copyright (C) 2017 Zachary Trefler and Atif Mahmud

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

If you have comments, questions, concerns, or you just want to say 'hi',
email Zachary Trefler at zmct99@gmail.com or Atif Mahmud at atifmahmud101@gmail.com

### Introduction:
Algal blooms, the sudden and rapid reproduction of microscopic algae or cyanobacteria in water bodies, have recently become a serious ecological issue threatening both the vitality of their local ecosystems and the socioeconomic conditions of their affected areas. These algal blooms, caused by eutrophication (excesses of nutrients in the water), often due to local farm fertilizer runoff, have the potential to engulf entire coastal regions. These enormous blooms deplete sunlight in the area, as well as absorbing large quantities of dissolved oxygen when they decay, depriving the local aquatic ecosystems of vital resources and causing them to collapse into dead zones. In addition, the algae within these blooms often produce toxic chemicals as by-products of their metabolism. These algal blooms, known as Harmful Algal Blooms or HABs, are becoming increasingly prevalent in today's society with the advent of modern agricultural techniques without sufficiently sophisticated environmental precautions to match. If the presence of algal blooms could be predicted, recovery efforts could be optimized to coincide with the algal blooms' initial outbreak, minimizing both the ecological and socioeconomic costs of the blooming. Even as the world of real-time oceanographic monitoring systems becomes more accessible and descriptive, early response efforts are still limited by the scope of the knowledge that these systems provide them; specifically, by the fact that all the information that these systems provide is real-time at best.

### Method:
We propose a method of predicting future HABs by taking advantage of possible patterns derived from past algal bloom events in conjunction with other oceanographic data. To do so, we have employed various machine learning approaches to define a model which scores the probability of a future algal bloom event from current data, similar to a meteorological projection. To generate our model, we used machine-learning neural techniques, experimenting with approaches such as genetic weight breeding, back-propagation., deep belief, etc. which analyzed past oceanographic data against algal blooms, and then developed a set of weights to match the data. The algorithm functioned as a neural network, with the input layer being the oceanographic data, the output layer the chance of an algal bloom. Our weighted synapses were manipulated with a combination of random generation and selective combination, based on the fitness of our output to the historical algal bloom data. In doing so, we discovered relationships between algal density and environmental parameters (such as water salinity; pH; chlorophyll-a concentration; silicates; chemical nutrients e.g. ammonia, phosphorous, and nitrogen; and water surface temperature).

### Findings and Next Steps:
Our algorithm has produced a model which matches our datasets enough to predict algal blooms within our data range, acceptably consistent with observed data. However, our algorithm is also far from a perfect prediction of the future; it is more of an experiment with neural network technology applied to this area. The logical next step for our algorithm would be to gather much more data, including data for locations worldwide, more extensive historical training data, data with more measured factors, and current real-time data. With any, or preferably, all of these extensions to our dataset, our neural network would have an enormously larger field of data from which to learn, and could potentially become both more versatile and more accurate. It would also be useful to consider other neural network permutations further, using larger and/or broader networks, different error functions, adjustment procedures, etc. In general, our approach still has room for improvement, especially concerning data breadth, but our model represents a useful and interesting method of predicting algal blooms worthy of further investigation.
