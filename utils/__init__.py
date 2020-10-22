def iterChunks(data, chunkSize):
    return (data[pos:pos + chunkSize] for pos in range(0, len(data), chunkSize))

def formatDict(data: dict):
    '; '.join(
        f'{key}={value}' for key, value in data.items()
    )
