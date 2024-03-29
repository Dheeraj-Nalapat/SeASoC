/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE404_Improper_Resource_Shutdown__w32CreateFile_fclose_67b.c
Label Definition File: CWE404_Improper_Resource_Shutdown__w32CreateFile.label.xml
Template File: source-sinks-67b.tmpl.c
*/
/*
 * @description
 * CWE: 404 Improper Resource Shutdown or Release
 * BadSource:  Open a file using CreateFile()
 * Sinks: fclose
 *    GoodSink: Close the file using CloseHandle()
 *    BadSink : Close the file using fclose()
 * Flow Variant: 67 Data flow: data passed in a struct from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <windows.h>

typedef struct _CWE404_Improper_Resource_Shutdown__w32CreateFile_fclose_67_struct_type
{
    HANDLE a;
} CWE404_Improper_Resource_Shutdown__w32CreateFile_fclose_67_struct_type;

#ifndef OMITBAD

void CWE404_Improper_Resource_Shutdown__w32CreateFile_fclose_67b_bad_sink(CWE404_Improper_Resource_Shutdown__w32CreateFile_fclose_67_struct_type my_struct)
{
    HANDLE data = my_struct.a;
    if (data != INVALID_HANDLE_VALUE)
    {
        /* FLAW: Attempt to close the file using fclose() instead of CloseHandle() */
        fclose((FILE *)data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G uses the BadSource with the GoodSink */
void CWE404_Improper_Resource_Shutdown__w32CreateFile_fclose_67b_goodB2G_sink(CWE404_Improper_Resource_Shutdown__w32CreateFile_fclose_67_struct_type my_struct)
{
    HANDLE data = my_struct.a;
    if (data != INVALID_HANDLE_VALUE)
    {
        /* FIX: Close the file using CloseHandle() */
        CloseHandle(data);
    }
}

#endif /* OMITGOOD */
