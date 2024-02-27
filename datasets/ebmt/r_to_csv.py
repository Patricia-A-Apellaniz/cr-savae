import pyreadr  # To read RData directly

# Load the RData file
for name in ['ebmt1', 'ebmt2', 'ebmt3', 'ebmt4']:
    rdata = pyreadr.read_r(name + '.RData')
    df = rdata[name]
    df.to_csv(name + '.csv', index=False)
print('Data processed')