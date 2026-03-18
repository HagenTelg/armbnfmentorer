import subprocess
import pathlib as pl
import xarray as xr
import pandas as pd
from armbnfmentorer.optional_imports import matplotlib, IPython, PIL, OptionalImport
if matplotlib.module_available:
    import matplotlib.pyplot as plt
import numpy as np
from tarfile import open as taropen
from io import BytesIO
# from PIL import Image
# from IPython.display import display
ipywidgets = OptionalImport("ipywidgets")

def rsync_bnfradsys(server: str = 'research.adc.arm.gov',
                    user_remote: str = 'hagentelg', 
                    path2localfld: str = "/Users/htelg/data/arm/datastream/bnf",
                    path2remote: list = ["/data/datastream/bnf/bnfradsys*", 
                                         "/data/datastream/bnf/bnfskyrad60sM1.b1", 
                                         "/data/datastream/bnf/bnfgndrad60sM1.b1"
                                         ],
                    verbose = False) -> None:
    """
    Note, this requires an installed ssh key here and on the remote system.
    Syncing the radsys, skyrad, and ground rad data from the bnf server with your local machine (no change on server, just copy it over).

    Parameters
    ----------
    user_remote : str
        Username for the remote server.
    path2localfld : str
        Local directory path to sync the data to.
    """
    local_dir = pl.Path(path2localfld)
    local_dir.mkdir(parents=True, exist_ok=True)

    # sources = [
    #     f"{user_remote}@research.adc.arm.gov:/data/datastream/bnf/bnfradsys*",
    #     f"{user_remote}@research.adc.arm.gov:/data/datastream/bnf/bnfskyrad60sM1.b1",
    #     f"{user_remote}@research.adc.arm.gov:/data/datastream/bnf/bnfgndrad60sM1.b1",
    # ]
    sources = [f"{user_remote}@{server}:{p}" for p in path2remote]
    if verbose:
        print(f"running the rsync command: rsync -avz -e 'ssh' {' '.join(sources)} {str(local_dir)}")
    subprocess.run(
        [
            "rsync",
            "-avz",
            "-e", "ssh",
            *sources,
            str(local_dir),
        ],
        check=True,
    )

def show_image_from_tar(tar_path, member_name=None, interactive=True):
    """This is designed to show images from the allsky camera tar files. It requires you to be on the ARM server, e.g. 
     via the jupyter-hub or you need to download all the data
     
     Parameters
     ----------
     tar_path : str
         Path to the tar file containing the images. E.g. '/data/archive/bnf/bnfasiskyimageS10.a1/bnfasiskyimageS10.a1.20250521.000000.jpg.tar'
     member_name : str, optional
         Specific member (image) to show from the tar file. If None, a Jupyter widget
         with previous/next arrows and a dropdown selector is shown when available.
     interactive : bool, optional
         If True and member_name is None, show interactive controls in Jupyter Lab.
         Default is True.
     """
    tf = taropen(tar_path, mode="r:*")
    image_members = [
        m.name
        for m in tf.getmembers()
        if m.isfile() and m.name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    ]
    if not image_members:
        tf.close()
        raise FileNotFoundError("No image file found in the tar.")

    image_cache = {}

    def _load_image(name):
        if name in image_cache:
            return image_cache[name]
        fobj = tf.extractfile(name)
        if fobj is None:
            raise FileNotFoundError(f"Could not open {name} from tar.")
        img = PIL.Image.open(BytesIO(fobj.read()))
        image_cache[name] = img
        return img

    def _display_image(name):
        img = _load_image(name)
        IPython.display.display(img)

    if member_name is not None:
        if member_name not in image_members:
            tf.close()
            raise FileNotFoundError(f"{member_name} not found in tar image list.")
        _display_image(member_name)
        tf.close()
        return member_name

    if not (interactive and ipywidgets.module_available and IPython.module_available):
        print("choose from list below")
        print("======================")
        for name in image_members:
            print(name)
        _display_image(image_members[0])
        tf.close()
        return image_members[0]

    widgets = ipywidgets
    dropdown = widgets.Dropdown(
        options=image_members,
        value=image_members[0],
        description="Image:",
        layout=widgets.Layout(width="85%"),
    )
    button_prev = widgets.Button(description="◀", tooltip="Previous image", layout=widgets.Layout(width="42px"))
    button_next = widgets.Button(description="▶", tooltip="Next image", layout=widgets.Layout(width="42px"))
    output = widgets.Output()

    def _render(name):
        with output:
            output.clear_output(wait=True)
            _display_image(name)

    def _move(step):
        idx = image_members.index(dropdown.value) + step
        idx = max(0, min(idx, len(image_members) - 1))
        dropdown.value = image_members[idx]

    def _on_dropdown_change(change):
        if change["name"] == "value" and change["type"] == "change":
            _render(change["new"])

    button_prev.on_click(lambda _: _move(-1))
    button_next.on_click(lambda _: _move(1))
    dropdown.observe(_on_dropdown_change, names="value")

    IPython.display.display(widgets.HBox([button_prev, button_next, dropdown]), output)
    _render(dropdown.value)

    # Return handles so callers can keep state alive and close tar explicitly when done.
    return {"tar": tf, "dropdown": dropdown, "prev": button_prev, "next": button_next, "output": output}

        
def get_file_availability(days = 7, stream = 'b1', base_path = '/Users/htelg/data/arm/datastream/bnf/',
              include_M1 = False, verbose = False,
             ):
    """
    Docstring for get_file_availability.

    Returns
    -------
    pd.DataFrame
        DataFrame indicating file availability for tower, ground, and M1 data.
        X indicates the presence of a file for the corresponding timestamp.
    """
    base_path = pl.Path(base_path)
    if stream == 'b1':
        p2fld_tower = base_path / 'bnfradsys43m60sS10.b1'
        p2fld_ground = base_path / 'bnfradsys2m60sS10.b1'
        if verbose:
            print(f'checking path {p2fld_tower} and {p2fld_ground}')
        add2M1 = pd.to_timedelta(30, 's')
    elif stream == 'a1':
        p2fld_tower = base_path / 'bnfradsys43mS10.a1'
        p2fld_ground = base_path / 'bnfradsys2mS10.a1'
        add2M1 = pd.to_timedelta(0, 's')
    elif stream == 'a0':
        p2fld_tower = base_path / 'bnfradsys43mS10.a0'
        p2fld_ground = base_path / 'bnfradsys2mS10.a0'
        add2M1 = pd.to_timedelta(0, 's')
    elif stream == '00':
        p2fld_tower = base_path / 'bnfradsys43mS10.00'
        p2fld_ground = base_path / 'bnfradsys2mS10.00'
        add2M1 = pd.to_timedelta(0, 's')
    else:
        assert(False), 'mooooeeeep'
    
    p2fld_M1 = base_path / 'bnfskyrad60sM1.b1'
    p2fld_M1g = base_path / 'bnfgndrad60sM1.b1'
    
    end = pd.Timestamp.now(tz = 'UTC').date()
    start = end - pd.to_timedelta(days, 'd')
    
    # get list of files and sort
    df_list = []
    
    files_tower = sorted(p2fld_tower.glob('*'), key=lambda f: '.'.join(f.name.split('.')[2:4]))
    index = pd.Series(files_tower).apply(lambda row: pd.to_datetime('.'.join(row.name.split('.')[2:4]), format = '%Y%m%d.%H%M%S')).dropna()
    df = pd.DataFrame(['X'] * index.shape[0], index = index, columns=['tower']).truncate(start, end)
    df_list.append(df)

    files_ground = sorted(p2fld_ground.glob('*'), key=lambda f: '.'.join(f.name.split('.')[2:4]))
    index = pd.Series(files_ground).apply(lambda row: pd.to_datetime('.'.join(row.name.split('.')[2:4]), format = '%Y%m%d.%H%M%S')).dropna()
    df_list.append(pd.DataFrame(['X'] * index.shape[0], index = index, columns=['ground']).truncate(start, end))
    if include_M1:
        files_M1 = sorted(p2fld_M1.glob('*.nc'), key=lambda f: '.'.join(f.name.split('.')[2:4]) )
        index = pd.Series(files_M1).apply(lambda row: pd.to_datetime('.'.join(row.name.split('.')[2:4]), format = '%Y%m%d.%H%M%S'))
        df_list.append(pd.DataFrame(['X'] * index.shape[0], index = index+add2M1, columns=['M1']).truncate(start, end))
        # files_M1g = sorted(p2fld_M1g.glob('*.nc'), key=lambda f: '.'.join(f.name.split('.')[2:4]) )

    #concat them all
    # return df_list
    out = pd.concat(df_list, axis = 1)
    return out

def plot_masked_clearsky(datac1, xlim_right_now = True):
    ds_tower = datac1['tower']
    
    f,a = plt.subplots()
    f.set_figwidth(f.get_figwidth() *1.5)
    alpha = 0.4
    lw = 1.5
    lw2 = 1
    g,=ds_tower.down_short_hemisp.plot(
                                        # marker = '.', 
                                        ls = '-', lw = lw2, 
                                        # markersize = 3, 
                                        alpha = alpha)
    g.set_markeredgewidth(0)
    col = g.get_color()
    gg = g
    
    g, = ds_tower.down_short_hemisp.where(ds_tower.mask_clear_sky_shortwave_radflux).plot(lw = lw)
    g.set_color(col)
    ###
    
    g,=ds_tower.down_short_diffuse_hemisp.plot(
                                        # marker = '.', 
                                        ls = '-', lw = lw2,
                                        # markersize = 3, 
                                        alpha = alpha)
    g.set_markeredgewidth(0)
    col = g.get_color()
    g, = ds_tower.down_short_diffuse_hemisp.where(ds_tower.mask_clear_sky_shortwave_radflux).plot(lw = lw)
    g.set_color(col)
    
    ###
    g,=ds_tower.down_short_direct_normal.plot(
                                        # marker = '.', 
                                        ls = '-', lw = lw2,
                                        # markersize = 3, 
                                        alpha = alpha)
    g.set_markeredgewidth(0)
    col = g.get_color()
    g, = ds_tower.down_short_direct_normal.where(ds_tower.mask_clear_sky_shortwave_radflux).plot(lw = lw)
    
    g.set_color(col)
    for dt in ds_tower.time.where(~ds_tower.mask_clear_sky_shortwave_radflux).dropna('time'):
        a.axvline(dt.values, color = '0.2', zorder = 0, lw = 0.01)
    
    y = gg.get_ydata()
    y = y[np.isfinite(y)]
    pad = (y.max() - y.min()) * 0.05
    a.set_ylim(y.min() - pad, y.max() + 4 * pad)
    if xlim_right_now:
        a.set_xlim(right = pd.Timestamp.now())
    return f,a

def plot_clearsky_masks(datac1):
    ds = datac1['tower']
    
    f,a = plt.subplots()
    masks = [
            'mask_clear_sky_shortwave_radflux',
            'mask_normalized_global_magnitude',
            'mask_diffuse_magnitude',
            'mask_global_irradiance_variability',
            'mask_normalized_diffuse_ratio_variability',
    ]
    for e,m in enumerate(masks):
        (ds[m] + e * 0.1).plot(label = m)
    a.legend(fontsize = 'xx-small')
    return f,a

def plot_housekeeping(data):
    merge_M1_skyrad_grdrad(data) # ensures that skyrad and grdrad are merged into M1, needed for qclib
    ds_tower = data['tower']
    ds_ground = data['ground']
    ds_M1 = data['M1']
    stream = data['stream']
    days = data['days']
    
    f,aa = plt.subplots(6, sharex=True, gridspec_kw={'hspace': 0})
    f.set_figheight(f.get_figheight() * 2)
    f.set_figwidth(f.get_figwidth() * 1.5)
    
    ##############################
    a = aa[0]
    if 'a' in stream:
        ds_tower.down_long_hemisp_vent_tachometer.plot(ax = a, label = 'tower-short')
        ds_tower.down_short_hemisp_vent_tachometer.plot(ax = a, label = 'tower-long')
    
        ds_ground.down_long_hemisp_vent_tachometer.plot(ax = a, label = 'ground-short')
        ds_ground.down_short_hemisp_vent_tachometer.plot(ax = a, label = 'ground-long')
    if 'b' in stream:
        text = 'No ventilation data in b1 data!'
        a.text(0.5, 0.5, text, transform = a.transAxes, ha = 'center')
    a.set_ylim(4500, 5500)
    a.set_ylabel('vent (rpm)')
    ##############################
    a = aa[1]
    # ds_tower.logger_volt_5v.plot(ax = a, label = 'tower-battery')
    ds_tower.loggerps_bat.plot(ax = a, label = 'logger battery')
    ds_tower.granite_ps.plot(ax = a, label = 'granite voltage')
    at = a.twinx()
    ds_tower.logger_volt_5v.plot(ax = at, label = 'logger-voltage', color = 'red')

    # if 1:
    #     ds_ground_sel.logger_volt_5v.plot(ax = a, label = 'ground-battery')
    #     ds_ground_sel.loggerps_bat.plot(ax = a, label = 'ground-supply')
    #     ds_ground_sel.granite_ps.plot(ax = a, label = 'ground-supply')
    a.legend()
    a.set_ylabel('voltage (V)')   
    at.set_ylabel('voltage (V)')

    ##############################
    a = aa[2]
    ds_tower.logger_temp.plot(ax = a, label = 'tower-l-temp')
    ds_tower.granite_temp1.plot(ax = a, label = 'tower-g-temp1')
    ds_tower.granite_temp2.plot(ax = a, label = 'tower-g-temp2')
    
    ds_ground.logger_temp.plot(ax = a, label = 'ground-l-temp')
    ds_ground.granite_temp1.plot(ax = a, label = 'ground-g-temp1')
    ds_ground.granite_temp2.plot(ax = a, label = 'ground-g-temp2')
    
    ##############################
    a = aa[3]
    ds_tower.inst_temp.plot(ax = a, label = 'tower')
    ds_ground.inst_temp.plot(ax = a, label = 'ground')
    a.legend()
    ##############################
    a = aa[4]
    ds_tower.inst_rh.plot(ax = a, label = 'tower')
    ds_ground.inst_rh.plot(ax = a, label = 'ground')
    a.legend()
    ###################################
    a = aa[5]
    ds_tower.clean_flag.plot(ax = a, label = 'tower')
    ds_ground.clean_flag.plot(ax = a, label = 'ground')
    a.legend()
    return f,aa


def plot_all_temperature_probes(dataa1):
    tmpvars =  ['inst_down_short_hemisp_case_temp',
                'inst_up_short_hemisp_case_temp',
                'inst_down_long_hemisp_case_temp',
                'inst_down_long_hemisp_dome_temp',
                'inst_up_long_hemisp_case_temp',
                'inst_up_long_hemisp_dome_temp',
                # 'inst_temp', #this is in C the others in K
               ]
    tmpvars.sort()
    
    
    f,aa =plt.subplots(2, gridspec_kw={'hspace': 0})
    for e,i in enumerate(['tower', 'ground']):
        a = aa[e]
        ds = dataa1[i]
        for var in tmpvars:
            vara = var.replace('inst_', '').replace('_temp','')
            ds[var].plot(ax = a, label = vara)
        
        a.legend(fontsize = 'x-small')
        a.set_ylabel(f'{i} temps [K]')
    return f,aa

def find_last_cleaning(radsys = 'tower', base_path = '/Users/htelg/data/arm/datastream/bnf/'):
# p2fld = pl.Path('/data/datastream/bnf/bnfradsys43m60sS10.b1/')
    if radsys == 'tower':
        p2fld = pl.Path(f'{base_path}') / 'bnfradsys43m60sS10.b1'
    elif radsys == 'ground':
        p2fld = pl.Path(f'{base_path}') / 'bnfradsys2m60sS10.b1'
    else:
        assert(False),'moooop'
    
    p2flist = list(p2fld.glob('*'))
    
    p2flist.sort(reverse=True)

    assert(len(p2flist) > 0), f'no files found in {p2fld}'
    for p2f in p2flist:
        ds = xr.open_dataset(p2f)
        if float(ds.clean_flag.sum()) > 0:
            break
    
    dtc = pd.to_datetime(p2f.name.split('.')[2])
    
    td = (pd.Timestamp.now(tz = 'UTC').tz_localize(None) - dtc)
    
    td.days
    
    print(f'last cleaning on {radsys} was {td.days} days ago; {dtc.date()}')
    
    ds.clean_flag.plot()
    return ds

def plot_spn1_vs_sr20(data):
    merge_M1_skyrad_grdrad(data) # ensures that skyrad and grdrad are merged into M1, needed for qclib
    ds_tower = data['tower']
    ds_ground = data['ground']
    ds_M1 = data['M1']
    stream = data['stream']
    days = data['days']
    f,aa = plt.subplots(6, sharex = True,height_ratios=[2,1,
                                                        2,1,
                                                        2,1,
                                                        # 2,1
                                                       ], gridspec_kw={'hspace': 0})
    f.set_figheight(f.get_figheight() * 1.5)
    f.set_figwidth(f.get_figwidth() * 1.5)
    mz_ratio = 0.1
    ####################
    a = aa[0]
    dst_tower = ds_tower['inst_down_short_hemisp' if 'a' in data['stream'] else 'down_short_hemisp']
    dst_tower.plot(ax = a, label = 'SR20')
    
    dst_tower_spn1 = ds_tower['inst_down_short_hemisp_spn1' if 'a' in data['stream'] else 'down_short_hemisp_spn1']
    dst_tower_spn1.plot(ax = a, label = 'SPN1', ls = '--', lw = 1)
    a.legend(title = 'tower', fontsize = 'small', loc = 1)
    now = pd.Timestamp.now(tz = 'UTC')
    a.set_xlim(now - pd.to_timedelta(days, 'd'), now)
    
    a = aa[1]
    (dst_tower_spn1/ dst_tower).plot(ax = a, label = 'tower', marker = '.', ls = '', markersize = mz_ratio)
    # (ds_tower.inst_down_short_hemisp_spn1 / ds_M1_itp.down_short_hemisp).plot(ax = a, label = 'tower', marker = '.', ls = '', markersize = mz_ratio)
    
    a.set_ylim(0,2)
    a.axhline(1, color = 'black', ls = '--', alpha = 0.5)
    
    a.set_ylabel('spn1/sr20')
    
    ####################
    a = aa[2]
    dst_ground = ds_ground['inst_down_short_hemisp' if 'a' in data['stream'] else 'down_short_hemisp']
    dst_ground.plot(ax = a, label = 'SR20')
    
    dst_ground_spn1 = ds_ground['inst_down_short_hemisp_spn1' if 'a' in data['stream'] else 'down_short_hemisp_spn1']
    dst_ground_spn1.plot(ax = a, label = 'SPN1', ls = '--', lw = 1)
    a.legend(title = 'ground', fontsize = 'small', loc = 1)
    now = pd.Timestamp.now(tz = 'UTC')
    a.set_xlim(now - pd.to_timedelta(days, 'd'), now)
    
    a = aa[3]
    (dst_ground_spn1/ dst_ground).plot(ax = a, label = 'tower', marker = '.', ls = '', markersize = mz_ratio)
    # (ds_tower.inst_down_short_hemisp_spn1 / ds_M1_itp.down_short_hemisp).plot(ax = a, label = 'tower', marker = '.', ls = '', markersize = mz_ratio)
    
    a.set_ylim(0,2)
    a.axhline(1, color = 'black', ls = '--', alpha = 0.5)
    
    a.set_ylabel('spn1/sr20')
    
    ####################
    a = aa[4]
    dst_ground = ds_ground['inst_down_short_hemisp' if 'a' in data['stream'] else 'down_short_hemisp']
    dst_ground.plot(ax = a, label = 'SR20')
    
    dst_ground_spn1 = ds_ground['inst_down_short_diffuse_hemisp_spn1' if 'a' in data['stream'] else 'down_short_diffuse_hemisp_spn1']
    dst_ground_spn1.plot(ax = a, label = 'SPN1 - diffuse', ls = '--', lw = 1)
    a.legend(title = 'ground', fontsize = 'small', loc = 1)
    now = pd.Timestamp.now(tz = 'UTC')
    a.set_xlim(now - pd.to_timedelta(days, 'd'), now)
    
    a = aa[5]
    (dst_ground_spn1/ dst_ground).plot(ax = a, label = 'tower', marker = '.', ls = '', markersize = mz_ratio)
    # (ds_tower.inst_down_short_hemisp_spn1 / ds_M1_itp.down_short_hemisp).plot(ax = a, label = 'tower', marker = '.', ls = '', markersize = mz_ratio)
    
    a.set_ylim(0,2)
    a.axhline(1, color = 'black', ls = '--', alpha = 0.5)
    
    a.set_ylabel('spn1/sr20')
    #######################3
    now = pd.Timestamp.now(tz = 'UTC')#p2fld_tower = pl.Path('/data/datastream/bnf/bnfradsys43mS10.a1/')
    
    a.set_xlim(now - pd.to_timedelta(days, 'd'), now)
    return f,aa

def plot_is_data_comming_in(data):
    merge_M1_skyrad_grdrad(data) # ensures that skyrad and grdrad are merged into M1, needed for qclib
    days = data['days']
    ds_tower = data['tower']
    f,aa = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0})
    a = aa[0]
    ds_tower['inst_down_short_hemisp' if 'a' in data['stream'] else 'down_short_hemisp'].plot(ax = a, label = 'tower')
    a.legend()
    
    a = aa[1]
    ds_ground = data['ground']
    ds_ground['inst_down_short_hemisp' if 'a' in data['stream'] else 'down_short_hemisp'].plot(ax = a, label = 'ground')
    a.legend()
    
    a = aa[2]
    ds_M1 = data['M1']
    ds_M1.down_short_hemisp.plot(ax = a, label = 'M1')
    a.legend()
    ########
    ### shading
    now = pd.Timestamp.now(tz = 'UTC')
    for a in aa:
        end = now.date()
        for e in range(days):
            start = end-pd.to_timedelta(1, 'd')
            col = str((e)%2) 
            # print(f'{e}, {start}, {end}, {col}')
            a.axvspan(start, end, color = col, alpha = 0.2)
            end = start
    #######################3
    a.set_xlim(now - pd.to_timedelta(days, 'd'), now)
    return f,aa

def plot_tower_vs_ground_down(data):
    ds_tower = data['tower']
    ds_ground = data['ground']
    # ds_M1 = data['M1']
    stream = data['stream']
    days = data['days']
    f,aa = plt.subplots(8, sharex = True,height_ratios=[2,1,2,1,2,1,2,1], gridspec_kw={'hspace': 0})
    f.set_figheight(f.get_figheight() * 2)
    f.set_figwidth(f.get_figwidth() * 1.5)
    mz_ratio = 0.1
    ####################
    a = aa[0]
    ds_tower['inst_down_short_hemisp' if 'a' in data['stream'] else 'down_short_hemisp'].plot(ax = a, label = 'tower')
    # ds_tower.inst_down_short_hemisp_spn1.plot(ax = a, label = 'tower SPN1')
    
    ds_ground['inst_down_short_hemisp' if 'a' in data['stream'] else 'down_short_hemisp'].plot(ax = a, label = 'ground', ls = '--', lw = 1)
    ds_ground['inst_down_short_hemisp_spn1' if 'a' in data['stream'] else 'down_short_hemisp_spn1'].plot(ax = a, label = 'ground SPN1')
    
    a.legend(title = 'down short global', fontsize = 'small', loc = 1)
    now = pd.Timestamp.now(tz = 'UTC')
    a.set_xlim(now - pd.to_timedelta(days, 'd'), now)
    
    a = aa[1]
    (ds_tower['inst_down_short_hemisp' if 'a' in data['stream'] else 'down_short_hemisp'] / ds_ground['inst_down_short_hemisp' if 'a' in data['stream'] else 'down_short_hemisp']).plot(ax = a, label = 'tower', marker = '.', ls = '', markersize = mz_ratio)
    (ds_tower['inst_down_short_hemisp' if 'a' in data['stream'] else 'down_short_hemisp'] / ds_ground['inst_down_short_hemisp' if 'a' in data['stream'] else 'down_short_hemisp_spn1']).plot(ax = a, label = 'tower', marker = '.', ls = '', markersize = mz_ratio)
    
    a.set_ylim(0.8,2.8)
    a.axhline(1, color = 'black', ls = '--', alpha = 0.5)
    ####################
    # down short diffuse
    a = aa[2]
    ds_tower['inst_down_short_diffuse_hemisp_spn1' if 'a' in data['stream'] else 'down_short_diffuse_hemisp_spn1'].plot(ax = a, label = 'tower SPN1')
    ds_ground['inst_down_short_diffuse_hemisp_spn1' if 'a' in data['stream'] else 'down_short_diffuse_hemisp_spn1'].plot(ax = a, label = 'ground', ls = '--')
    a.legend(title = 'down short diffuse', fontsize = 'small', loc = 1)
    
    a = aa[3]
    (ds_tower['inst_down_short_diffuse_hemisp_spn1' if 'a' in data['stream'] else 'down_short_diffuse_hemisp_spn1'] / ds_ground['inst_down_short_diffuse_hemisp_spn1' if 'a' in data['stream'] else 'down_short_diffuse_hemisp_spn1']).plot(ax = a, marker = '.', ls = '', markersize = mz_ratio)
    a.set_ylim(0.8,2.6)
    a.axhline(1, color = 'black', ls = '--', alpha = 0.5)
    ####################
    # down short direct normal
    a = aa[4]
    dst_tower = ds_tower['inst_down_short_hemisp' if 'a' in data['stream'] else 'down_short_hemisp'] - ds_tower['inst_down_short_diffuse_hemisp_spn1' if 'a' in data['stream'] else 'down_short_diffuse_hemisp_spn1']
    dst_tower.plot(ax = a, label = 'tower SPN1')
    dst_ground = ds_ground['inst_down_short_hemisp' if 'a' in data['stream'] else 'down_short_hemisp'] - ds_ground['inst_down_short_diffuse_hemisp_spn1' if 'a' in data['stream'] else 'down_short_diffuse_hemisp_spn1']
    dst_ground.plot(ax = a, label = 'ground', ls = '--')
    a.legend(title = 'down short direct h.', fontsize = 'small', loc = 1)
    
    a = aa[5]
    (dst_ground/dst_tower).plot(ax = a, marker = '.', ls = '', markersize = mz_ratio)
    a.set_ylim(0,2)
    a.axhline(1, color = 'black', ls = '--', alpha = 0.5)
    ####################
    ## longwave
    a = aa[6]
    ds_tower['inst_down_long_hemisp' if 'a' in data['stream'] else 'down_long_hemisp'].plot(ax = a, label = 'tower')
    ds_ground['inst_down_long_hemisp' if 'a' in data['stream'] else 'down_long_hemisp'].plot(ax = a, label = 'ground', ls = '--')
    a.legend(title = 'down long', fontsize = 'small', loc = 1)
    
    a = aa[7]
    (ds_tower['inst_down_long_hemisp' if 'a' in data['stream'] else 'down_long_hemisp'] / ds_ground['inst_down_long_hemisp' if 'a' in data['stream'] else 'down_long_hemisp']).plot(ax = a, marker = '.', ls = '', markersize = mz_ratio)
    a.set_ylim(0,2)
    a.axhline(1, color = 'black', ls = '--', alpha = 0.5)
    #######################3
    now = pd.Timestamp.now(tz = 'UTC')
    a.set_xlim(now - pd.to_timedelta(days, 'd'), now)
    return f,aa
    
def plot_tower_vs_ground_up(data):
    merge_M1_skyrad_grdrad(data) # ensures that skyrad and grdrad are merged into M1, needed for qclib
    ds_tower = data['tower']
    ds_ground = data['ground']
    # ds_M1 = data['M1']
    stream = data['stream']
    days = data['days']
    f,aa = plt.subplots(4, sharex = True,height_ratios=[2,1,2,1,
                                                        # 2,1,2,1
                                                       ], gridspec_kw={'hspace': 0})
    f.set_figheight(f.get_figheight() * 1.5)
    f.set_figwidth(f.get_figwidth() * 1.5)
    mz_ratio = 0.1
    ####################
    a = aa[0]
    dst_tower = ds_tower['inst_up_short_hemisp' if 'a' in data['stream'] else 'up_short_hemisp']
    dst_tower.plot(ax = a, label = 'tower')
    
    dst_ground = ds_ground['inst_up_short_hemisp' if 'a' in data['stream'] else 'up_short_hemisp']
    dst_ground.plot(ax = a, label = 'ground', ls = '--', lw = 1)
    a.legend(title = 'up short global', fontsize = 'small', loc = 1)
    now = pd.Timestamp.now(tz = 'UTC')
    a.set_xlim(now - pd.to_timedelta(days, 'd'), now)
    
    a = aa[1]
    (dst_tower / dst_ground).plot(ax = a, label = 'tower', marker = '.', ls = '', markersize = mz_ratio)
    # (ds_tower.inst_down_short_hemisp_spn1 / ds_M1_itp.down_short_hemisp).plot(ax = a, label = 'tower', marker = '.', ls = '', markersize = mz_ratio)
    
    a.set_ylim(0,2)
    a.axhline(1, color = 'black', ls = '--', alpha = 0.5)
    
    ####################
    ## longwave
    a = aa[2]
    dst_tower = ds_tower['inst_up_long_hemisp' if 'a' in data['stream'] else 'up_long_hemisp']
    dst_tower.plot(ax = a, label = 'tower')
    dst_ground = ds_ground['inst_up_long_hemisp' if 'a' in data['stream'] else 'up_long_hemisp']
    dst_ground.plot(ax = a, label = 'ground', ls = '--')
    a.legend(title = 'up long', fontsize = 'small', loc = 1)
    
    a = aa[3]
    (dst_tower / dst_ground).plot(ax = a, marker = '.', ls = '', markersize = mz_ratio)
    a.set_ylim(0.9,1.1)
    a.axhline(1, color = 'black', ls = '--', alpha = 0.5)
    #######################3
    now = pd.Timestamp.now(tz = 'UTC')
    a.set_xlim(now - pd.to_timedelta(days, 'd'), now)
    return f,aa

def plot_downwelling(data):
    merge_M1_skyrad_grdrad(data) # ensures that skyrad and grdrad are merged into M1, needed for qclib
    ds_tower = data['tower']
    ds_ground = data['ground']
    ds_M1 = data['M1']
    stream = data['stream']
    days = data['days']
    f,aa = plt.subplots(8, sharex = True,height_ratios=[2,1,2,1,2,1,2,1], gridspec_kw={'hspace': 0})
    f.set_figheight(f.get_figheight() * 2)
    f.set_figwidth(f.get_figwidth() * 1.5)
    mz_ratio = 0.1
    ####################
    ### down short global
    a = aa[0]
    ds_tower['inst_down_short_hemisp' if 'a' in data['stream'] else 'down_short_hemisp'].plot(ax = a, label = 'tower')
    ds_tower['inst_down_short_hemisp_spn1' if 'a' in data['stream'] else 'down_short_hemisp_spn1'].plot(ax = a, label = 'tower SPN1')
    
    ds_M1.down_short_hemisp.plot(ax = a, label = 'M1', ls = '--', lw = 1)
    a.legend(title = 'down short global', fontsize = 'small', loc = 2)
    now = pd.Timestamp.now(tz = 'UTC')
    a.set_xlim(now - pd.to_timedelta(days, 'd'), now)
    a.set_ylabel('shortwave - global')
    
    a = aa[1]
    (ds_tower['inst_down_short_hemisp' if 'a' in data['stream'] else 'down_short_hemisp'] / ds_M1.down_short_hemisp).plot(ax = a, label = 'tower', marker = '.', ls = '', markersize = mz_ratio)
    (ds_tower['inst_down_short_hemisp_spn1' if 'a' in data['stream'] else 'down_short_hemisp_spn1'] / ds_M1.down_short_hemisp).plot(ax = a, label = 'tower', marker = '.', ls = '', markersize = mz_ratio)
    
    a.set_ylim(0,2)
    a.axhline(1, color = 'black', ls = '--', alpha = 0.5)
    a.set_ylabel('ratio')
    ####################
    # down short diffuse
    a = aa[2]
    ds_tower['inst_down_short_diffuse_hemisp_spn1' if 'a' in data['stream'] else 'down_short_diffuse_hemisp_spn1'].plot(ax = a, label = 'tower SPN1')
    ds_M1.down_short_diffuse_hemisp.plot(ax = a, label = 'M1', ls = '--')
    a.legend(title = 'down short diffuse', fontsize = 'small', loc = 2)
    a.set_ylabel('shortwave - diffuse')
    
    a = aa[3]
    (ds_tower['inst_down_short_diffuse_hemisp_spn1' if 'a' in data['stream'] else 'down_short_diffuse_hemisp_spn1'] / ds_M1.down_short_diffuse_hemisp).plot(ax = a, marker = '.', ls = '', markersize = mz_ratio)
    a.set_ylim(0,2)
    a.axhline(1, color = 'black', ls = '--', alpha = 0.5)
    a.set_ylabel('ratio')
    
    ####################
    # down short direct normal
    a = aa[4]
    if 'a' in stream:
        dst_tower = ds_tower.inst_down_short_hemisp - ds_tower.inst_down_short_diffuse_hemisp_spn1
        dst_tower.plot(ax = a, label = 'tower SPN1')
    else:
        dst_tower = ds_tower.down_short_direct_hemisp
        ds_tower.down_short_direct_hemisp.plot(ax = a, label = 'tower (global - diffuse_SPN1)')
        
    dst_m1 = ds_M1.down_short_hemisp - ds_M1.down_short_diffuse_hemisp
    dst_m1.plot(ax = a, label = 'M1', ls = '--')
    a.legend(title = 'down short direct h.', fontsize = 'small', loc = 2)
    a.set_ylabel('shortwave - direct')
    
    a = aa[5]
    dst_m1 = ds_M1.down_short_hemisp - ds_M1.down_short_diffuse_hemisp
    (dst_m1/dst_tower).plot(ax = a, marker = '.', ls = '', markersize = mz_ratio)
    a.set_ylim(0,2)
    a.axhline(1, color = 'black', ls = '--', alpha = 0.5)
    a.set_ylabel('ratio')
    ####################
    ## longwave
    a = aa[6]
    ds_tower['inst_down_long_hemisp' if 'a' in data['stream'] else 'down_long_hemisp'].plot(ax = a, label = 'tower')
    ds_M1.down_long_hemisp1.plot(ax = a, label = 'M1', ls = '--')
    a.legend(title = 'down long', fontsize = 'small', loc = 2)
    a.set_ylabel('longwave')
    
    a = aa[7]
    (ds_tower['inst_down_long_hemisp' if 'a' in data['stream'] else 'down_long_hemisp'] / ds_M1.down_long_hemisp1).plot(ax = a, marker = '.', ls = '', markersize = mz_ratio)
    a.set_ylim(0,2)
    a.axhline(1, color = 'black', ls = '--', alpha = 0.5)
    a.set_ylabel('ratio')
    #######################3
    now = pd.Timestamp.now(tz = 'UTC')
    a.set_xlim(now - pd.to_timedelta(days, 'd'), now)
    
    # for e,a in enumerate(aa):
    #     if e%2 == 1:
    #         a.set_ylim(0.5, 1.5)
    #     else:
    #         leg = a.legend(loc = 2)
        
    return f,aa


def plot_shadowing_test(datab1):
    ds = datab1['tower']
    
    start = pd.to_datetime(ds.time.values[0])
    start = pd.to_datetime(start.date()) + pd.to_timedelta(6, 'h')
    
    f,aa = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0})
    alpha = 0.8
    while start < ds.time.values[-1]:
        end = start + pd.to_timedelta(24, 'h')
        dst = ds.sel(time = slice(start,end))
        dt = pd.to_datetime(dst.time) - pd.to_datetime(start.date())
        dst['dt'] = ('time', dt)
        dst = dst.swap_dims({'time':'dt'})
        a = aa[0]
        dst.down_short_hemisp.plot(ax = a, label = start.date(), alpha = alpha)
        a = aa[1]
        dst.down_short_direct_hemisp.plot(ax = a, label = start.date(), alpha = alpha)
        a = aa[2]
        (dst.down_short_hemisp - dst.down_short_hemisp_spn1).plot(ax = a, label = start.date(), alpha = alpha)
        start = end
    a.legend()
    return f,aa

def merge_M1_skyrad_grdrad(data):   
    if 'M1' in data:
        return
    ds_M1 = data['M1_skyrad']
    ds_M1g = data['M1_grdrad']
    ds = data['tower']
    ds_M1 = xr.merge([ds_M1, ds_M1g], compat='override')
    # often the time stemps are not identical -> interpolate 
    ds_M1 = ds_M1.interp(time = ds.time).compute()
    data['M1'] = ds_M1
    return 

def plot_upwelling_short_tower_vs_M1(data, ax = None):
    merge_M1_skyrad_grdrad(data) # ensures that skyrad and grdrad are merged into M1, needed for qclib
    if isinstance(ax, type(None)):
        f,a = plt.subplots()
    else:
        a = ax
        f = ax.figure

    ds_tower = data['tower']
    ds_M1 = data['M1']

    ds_tower['inst_up_short_hemisp' if 'a' in ds_tower.data_level else 'up_short_hemisp'].plot(ax = a, label = 'tower')
    ds_M1.up_short_hemisp.plot(ax = a, label = 'M1', ls = '--', lw = 1)
    a.legend(title = 'up short global', fontsize = 'small', loc = 1)
    return f,a



def plot_upwelling(data):
    merge_M1_skyrad_grdrad(data) # ensures that skyrad and grdrad are merged into M1, needed for qclib
    ds_tower = data['tower']
    ds_ground = data['ground']
    ds_M1 = data['M1']
    stream = data['stream']
    days = data['days']
    f,aa = plt.subplots(4, sharex = True,height_ratios=[2,1,2,1,
                                                        # 2,1,2,1
                                                       ], gridspec_kw={'hspace': 0})
    f.set_figheight(f.get_figheight() * 1.5)
    f.set_figwidth(f.get_figwidth() * 1.5)
    mz_ratio = 0.1
    ####################
    # down short global
    a = aa[0]
    plot_upwelling_short_tower_vs_M1(data, ax = a)
    now = pd.Timestamp.now(tz = 'UTC')
    a.set_xlim(now - pd.to_timedelta(days, 'd'), now)
    
    # 
    a = aa[1]
    (ds_tower['inst_up_short_hemisp' if 'a' in data['stream'] else 'up_short_hemisp'] / ds_M1.up_short_hemisp).plot(ax = a, label = 'tower', marker = '.', ls = '', markersize = mz_ratio)
    # (ds_tower.inst_down_short_hemisp_spn1 / ds_M1_itp.down_short_hemisp).plot(ax = a, label = 'tower', marker = '.', ls = '', markersize = mz_ratio)
    
    a.set_ylim(0,2)
    a.axhline(1, color = 'black', ls = '--', alpha = 0.5)
    
    ####################
    ## longwave
    a = aa[2]
    ds_tower['inst_up_long_hemisp' if 'a' in data['stream'] else 'up_long_hemisp'].plot(ax = a, label = 'tower')
    ds_M1.up_long_hemisp.plot(ax = a, label = 'M1', ls = '--')
    a.legend(title = 'up long', fontsize = 'small', loc = 1)
    
    a = aa[3]
    (ds_tower['inst_up_long_hemisp' if 'a' in data['stream'] else 'up_long_hemisp'] / ds_M1.up_long_hemisp).plot(ax = a, marker = '.', ls = '', markersize = mz_ratio)
    a.set_ylim(0,2)
    a.axhline(1, color = 'black', ls = '--', alpha = 0.5)
    #######################3
    now = pd.Timestamp.now(tz = 'UTC')
    a.set_xlim(now - pd.to_timedelta(days, 'd'), now)
    return f,aa

def load_data(days = 7, start = None, end = None, stream = 'b1', 
              include_M1_skyrad = True, 
              include_M1_groundrad = True, 
              include_radsys_tower = True, 
              include_radsys_ground = True, 
              source = 'datastream', 
              base_path = '/Users/htelg/data/arm/', 
              version = '0.1',
              verbose = False):
    """
    Parameters
    ----------
    days: int
        Number of days from start. If start is None, days gives number of days from now backwards. If start and end are given, this kwarg is ignored
    start: str
        Start date, e.g.'2025-09-16'
    end: str
        End date, e.g. '2025-09-16'
    source: str, ['datastream', 'archive']
        What folder to use, the data
    version: str
        Version of the data, e.g. 'v0.1'. Only relevant for value added products (stream 'c1').
    base_path: str
        Base path to data folder. On arm servers this is /data/ 
    """
    def p2fld2fileseries(p2fld):
        sr = pd.Series(p2fld.glob('*.nc'))
        if verbose:
            print(f'found {len(sr)} files in {p2fld}')

        def extract_date_from_row(row):
            nsplit = row.name.split('.')
            if nsplit[1] == 'c1':
                return pd.to_datetime(nsplit[-2])
            else:
                return pd.to_datetime(' '.join(nsplit[2:4]))
        sr.index = sr.apply(extract_date_from_row)
        sr.sort_index(inplace=True)
        srt = sr.truncate(start, end)
        if verbose:
            print(f'selected {len(srt)} files between {start} and {end}')
            if len(srt) == 0:
                print(f'dateformating wrong? fist index: {sr.index[0]}, from filename: {sr.iloc[0].name}')
        return srt
        
    out = dict(#tower = ds_tower,
           # ground = ds_ground,
           # M1 = ds_M1,
           stream = stream,
           days = days,
           # M1_down = ds_M1,
           # M1_up = ds_M1g,
          )
    if isinstance(start, type(None)):
        start = pd.Timestamp.now(tz = 'UTC').tz_localize(None) - pd.to_timedelta(days, 'd')
    else:
        if isinstance(end, type(None)):
            end = pd.to_datetime(start) + pd.to_timedelta(days, 'd')
    
    if stream == 'b1':
        p2fld_tower = pl.Path(f'{base_path}/{source}/bnf/bnfradsys43m60sS10.b1/')
        p2fld_ground = pl.Path(f'{base_path}/{source}/bnf/bnfradsys2m60sS10.b1/')
    elif stream == 'a1':
        p2fld_tower = pl.Path(f'{base_path}/{source}/bnf/bnfradsys43mS10.a1/')
        p2fld_ground = pl.Path(f'{base_path}/{source}/bnf/bnfradsys2mS10.a1/')
    elif stream == 'c1':
        p2fld_tower = pl.Path(f'{base_path}/bnfradsys43m60sS10.c1/{version}/')
        p2fld_ground = pl.Path(f'{base_path}/bnfradsys2m60sS10.c1/{version}/')
    else:
        assert(False), f'stream not recognised. Is: "{stream}"'
    
    p2fld_M1 = pl.Path(f'{base_path}/{source}/bnf/bnfskyrad60sM1.b1/')
    p2fld_M1g = pl.Path(f'{base_path}/{source}/bnf/bnfgndrad60sM1.b1/')
    
    

    
    if include_radsys_ground:
        if verbose:
            print(f'loading ground path: {p2fld_ground})')
        files_ground = p2fld2fileseries(p2fld_ground)
        if verbose:
            print(f'loading ground files: {files_ground})')
        ds = xr.open_mfdataset(files_ground)
        out['ground'] = ds
    if include_radsys_tower:
        if verbose:
            print(f'loading tower path: {p2fld_tower})')
        files_tower = p2fld2fileseries(p2fld_tower)
        if verbose:
            print(f'loading tower files: {files_tower})')
        try:
            ds = xr.open_mfdataset(files_tower, 
                                combine="nested",
                                concat_dim="time",  
                                #    coords='minimal',
                                )
        except ValueError as e:
            # try:
            print(f'error loading tower files with xarray, trying with coords="minimal": {e}')
            ds = xr.open_mfdataset(files_tower, 
                                combine="nested",
                                concat_dim="time",  
                                   coords='minimal',
                                )
            

        out['tower'] = ds
    if include_M1_skyrad:
        files_M1 = p2fld2fileseries(p2fld_M1)
        ds_M1 = xr.open_mfdataset(files_M1)
        out['M1_skyrad'] = ds_M1

    if include_M1_groundrad:
        files_M1g = p2fld2fileseries(p2fld_M1g)
        ds_M1g = xr.open_mfdataset(files_M1g)
        out['M1_grdrad'] = ds_M1g

        
    # get list of files and sort
    # files_tower = sorted(p2fld_tower.glob('*.nc'), key=lambda f: '.'.join(f.name.split('.')[2:4]))
    # files_ground = sorted(p2fld_ground.glob('*.nc'), key=lambda f: '.'.join(f.name.split('.')[2:4]))
    # files_M1 = sorted(p2fld_M1.glob('*.nc'), key=lambda f: '.'.join(f.name.split('.')[2:4]) )
    # files_M1g = sorted(p2fld_M1g.glob('*.nc'), key=lambda f: '.'.join(f.name.split('.')[2:4]) )

    # return files_tower, files_ground, files_M1, files_M1g
    # load the data
    # files_tower_sel = files_tower[-days:]
    # ds_tower = xr.open_mfdataset(files_tower)
    # files_ground_sel = files_ground[-days:]
    # ds_ground = xr.open_mfdataset(files_ground)
    # files_M1_sel = files_M1[-days:]
    # ds_M1 = xr.open_mfdataset(files_M1)
    # ds_M1g = xr.open_mfdataset(files_M1g)
    # ds_M1 = xr.merge([ds_M1, ds_M1g], compat='override')
    
    # often the time stemps are not identical -> interpolate 
    # ds_M1 = ds_M1.interp(time = ds_tower.time).compute()
    

    return out
