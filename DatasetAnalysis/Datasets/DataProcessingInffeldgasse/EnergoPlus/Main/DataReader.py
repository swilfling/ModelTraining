from pathlib import Path
from parse_excel_files import parse_excel_files

if __name__ == "__main__":

    extension = ".xls"
    root_dir = (Path(__file__).parent).parent.parent.parent
    WorkingDirectory = root_dir / "Data" / "Inffeldgasse" / "Data"
    print(WorkingDirectory)

    all_data = parse_excel_files(WorkingDirectory, extension)

    all_data.to_csv(WorkingDirectory / 'AllSensors.csv', index=True, index_label='Timestamp')

    #'''Visualize the missigness'''
    #we need to change the index format otherwise seaborn gives UTC000000 length labels.
#    all_data_visual = all_data.set_index(all_data.index.strftime('%d.%m.%Y'))
#    ax = sns.heatmap(all_data_visual.isna(), xticklabels=1, cbar=False)
#    ax.set_ylabel('Time Period', fontsize=20)
#    ax.set_xlabel('Sensors', fontsize=20)
#    ax.set_xticklabels(ax.get_xticklabels(), rotation=75, fontsize=15)
#    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)







    # This is now done in CSV Reader!

    # Drop Nans that are common in all columns
#    new_all_data = all_data.dropna(axis = 0, how = 'all')

#    '''Visualize the missigness Again'''
    #we need to change the index format otherwise seaborn gives UTC000000 length labels.
#    new_all_data_visual = new_all_data.set_index(new_all_data.index.strftime('%d.%m.%Y'))
#    ax = sns.heatmap(new_all_data_visual.isna(), xticklabels=1, cbar=False)
#    ax.set_ylabel('Time Period', fontsize=20)
#    ax.set_xlabel('Sensors', fontsize=20)
#    ax.set_xticklabels(ax.get_xticklabels(), rotation=75, fontsize=15)
#    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
#    new_all_data_visual.to_csv('AllNewSensors.csv', index=True)






