/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE404_Improper_Resource_Shutdown__freopen_w32CloseHandle_51b.c
Label Definition File: CWE404_Improper_Resource_Shutdown.label.xml
Template File: source-sinks-51b.tmpl.c
*/
/*
 * @description
 * CWE: 404 Improper Resource Shutdown or Release
 * BadSource: freopen Open a file using freopen()
 * Sinks: w32CloseHandle
 *    GoodSink: Close the file using fclose()
 *    BadSink : Close the file using CloseHandle
 * Flow Variant: 51 Data flow: data passed as an argument from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <windows.h>

#ifndef OMITBAD

void CWE404_Improper_Resource_Shutdown__freopen_w32CloseHandle_51b_bad_sink(FILE * data)
{
    if (data != NULL)
    {
        /* FLAW: Attempt to close the file using CloseHandle() instead of fclose() */
        CloseHandle((HANDLE)data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

void CWE404_Improper_Resource_Shutdown__freopen_w32CloseHandle_51b_goodB2G_sink(FILE * data)
{
    if (data != NULL)
    {
        /* FIX: Close the file using fclose() */
        fclose(data);
    }
}

#endif /* OMITGOOD */
