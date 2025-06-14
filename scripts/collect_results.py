import numpy as np 
import pdb

def tabilize(results, precisions, rank_order, suffixes=None, hlines = []):

  def rankify(x, order):
    # Turn a vector of values into a list of ranks, while handling ties.
    assert len(x.shape) == 1
    if order == 0:
      return np.full_like(x, 1e5, dtype=np.int32)
    u = np.sort(np.unique(x))
    if order == 1:
      u = u[::-1]
    r = np.zeros_like(x, dtype=np.int32)
    for ui, uu in enumerate(u):
      mask = x == uu
      r[mask] = ui
    return np.int32(r)

  names = results.keys()
  data = np.array(list(results.values()))
  assert len(names) == len(data)
  data = np.array(data)

  tags = [' \cellcolor{tabfirst}',
          '\cellcolor{tabsecond}',
          ' \cellcolor{tabthird}',
          '                     ']

  max_len = max([len(v) for v in list(names)])
  names_padded = [v + ' '*(max_len-len(v)) for v in names]

  data_quant = np.round((data * 10.**(np.array(precisions)[None, :]))) / 10.**(np.array(precisions)[None, :])
  if suffixes is None:
    suffixes = [''] * len(precisions)

  tagranks = []
  for d in range(data_quant.shape[1]):
    tagranks.append(np.clip(rankify(data_quant[:,d], rank_order[d]), 0, len(tags)-1))
  tagranks = np.stack(tagranks, -1)

  for i_row in range(len(names)):
    line = ''
    if i_row in hlines:
      line += '\\hline\n'
    line += names_padded[i_row]
    for d in range(data_quant.shape[1]):
      line += ' & '
      if rank_order[d] != 0 and not np.isnan(data[i_row,d]):
        line += tags[tagranks[i_row, d]]
      if np.isnan(data[i_row,d]):
        line += ' - '
      else:
        assert precisions[d] >= 0
        line += ('{:' + f'0.{precisions[d]}f' + '}').format(data_quant[i_row,d]) + suffixes[d]
    if i_row < len(names)-1:
      line += ' \\\\'
    print(line)
    
def load_results_file(path):
    # Load the data from the text file
    with open(path, 'r') as file:
        data = file.read()

    # Split data into lines
    lines = data.strip().split("\n")

    # Create a dictionary by parsing each line
    result = {}
    for line in lines:
        key, value = line.split(": ", 1)
        result[key.strip()] = eval(value)
    return result


def sim_results_table():
    checkpoint_path = "/scratch/ondemand28/anagh/active-yobo/checkpoints/yobo_results/synthetic"
    scenes = [
        "pots", 
        "cornell",
        "peppers", 
        "kitchen"
              ]
    methods = {"tnerf_cache":'T-NeRF', "cache":"FWP", "material_light_from_scratch_resample":"ours"}
    metrics = ["psnr", "lpips", "ssim", "mae", "l1_median", "transient_iou"]
    precisions = [2]*len(metrics)  # How many digits of precision to use.
    rank_order = [1, -1,1, -1, -1, 1]  # +1 = higher is better, -1 = lower is better, 0 = do not color code.
    suffixes = ['']*len(metrics)  # What string to append after each number.

    results = {}
    
    for method in methods.keys():
        results_list = [0]*len(metrics)
        for scene in scenes:
            try:
                results_path = f"{checkpoint_path}/{scene}_{method}_albedo/save/results.txt"
                res_dict = load_results_file(results_path)
                for ind, metric in enumerate(metrics):
                    results_list[ind] += res_dict[metric][-1]
            except Exception as e: 
                print(e)
                print("\n")
                print(f"No {results_path} \n")
        
        results_list = [x/len(scenes) for x in results_list]
        results[methods[method]] = results_list
        
            
    # results = {
    #     'Transient NeRF': [30.52436, 0.151243, 9.61],
    #     'FWP': [32.1315, 0.074125, 100.1231],
    #     'Ours': [19.26456, 0.43312, 3.10]}
    
    hlines = [] # Where to insert horizontal lines.
    tabilize(results, precisions, rank_order, suffixes=suffixes, hlines=hlines)


def per_scene_results_table():
    checkpoint_path = "/scratch/ondemand28/anagh/active-yobo/checkpoints/yobo_results/synthetic"
    scenes = [
        "pots", 
        # "cornell",
        # "peppers", 
        # "kitchen"
              ]
    
    methods = {"tnerf_cache":'T-NeRF', "cache":"FWP", "material_light_from_scratch_resample":"ours"}
    metrics = ["psnr", "lpips", "ssim", "mae", "l1_median", "transient_iou"]
    precisions = [2]*len(metrics)  # How many digits of precision to use.
    rank_order = [1, -1,1, -1, -1, 1]  # +1 = higher is better, -1 = lower is better, 0 = do not color code.
    suffixes = ['']*len(metrics)  # What string to append after each number.

    results = {}
    
    for method in methods.keys():
        results_list = [0]*len(metrics)
        for scene in scenes:
            try:
                results_path = f"{checkpoint_path}/{scene}_{method}_albedo/save/results.txt"
                res_dict = load_results_file(results_path)
                for ind, metric in enumerate(metrics):
                    results_list[ind] += res_dict[metric][-1]
            except Exception as e: 
                print(e)
                print("\n")
                print(f"No {results_path} \n")
        
        results_list = [x/len(scenes) for x in results_list]
        results[methods[method]] = results_list
        
            
    # results = {
    #     'Transient NeRF': [30.52436, 0.151243, 9.61],
    #     'FWP': [32.1315, 0.074125, 100.1231],
    #     'Ours': [19.26456, 0.43312, 3.10]}
    
    hlines = [] # Where to insert horizontal lines.
    tabilize(results, precisions, rank_order, suffixes=suffixes, hlines=hlines)


if __name__=="__main__":
    sim_results_table()
