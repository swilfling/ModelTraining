import os

import pandas as pd


def parse_excel_files(WorkingDirectory, extension):
    filenames = os.listdir(WorkingDirectory)
    print(filenames)

    all_data = pd.DataFrame()
    # Assigns which label offers data in what resolution
    # gather those sensors in the same folder
    for filename in filenames:
        if filename.endswith(extension):
            print(filename)
            full_xls_path = os.path.join(WorkingDirectory, filename)
            xls = pd.ExcelFile(full_xls_path)
            for sheet in xls.sheet_names:
                print(f'{sheet}')
                df = pd.read_excel(xls, header=3, sheet_name=sheet)
                df = df.set_index(pd.to_datetime(df['Datum'] + ' ' + df['Zeit'], format='%d.%m.%Y %H:%M:%S'))
                # At this stage you can save every sheet individual csv.
                # df.to_csv(sheet+'.csv', index=True)
                # Sensor units
                unit_column_name = "Einheit"
                unitSensor = df[unit_column_name][df[unit_column_name].notna()][0]  # take the unit
                df = df.rename(columns={'Wert': sheet + '_' + unitSensor},
                               inplace=False)  # assign columns Wert the codename and the unit
                df = df.drop(['Datum', 'Zeit', unit_column_name], axis=1)
                all_data[sheet + '_' + unitSensor] = df[sheet + '_' + unitSensor]

    return all_data