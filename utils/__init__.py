def iterChunks(data, chunkSize):
    return (data[pos:pos + chunkSize] for pos in range(0, len(data), chunkSize))
