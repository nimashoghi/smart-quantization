Get-ChildItem ./lightning_logs_feb5_bert -Directory | Where-Object { $_.Name -like "*smartfp*" } | ForEach-Object {
    $dict = @{}
    foreach ($item in Import-Csv "$_/version_0/meta_tags.csv") {
        $dict[$item.key] = $item.value
    }
    $tag = "$($dict['num_bits_main']),$($dict['num_bits_outlier'])"
    $parts = $_.Name.Split("-")
    $secondPart = $parts[$parts.Length - 1]
    $firstPart = ((0..($parts.Length - 2)) | ForEach-Object { $parts[$_] }) -join '-'

    echo"$($_.Parent.FullName)$firstPart-$tag-$secondPart"
}


$len = 'lightning_logs_feb5_bert'.Length
ls lightning_logs_feb5_bertsma* | ForEach-Object { Move-Item $_ "./lightning_logs_feb5_bert/$($_.Name.Substring('lightning_logs_feb5_bert'.Length)-Force }
