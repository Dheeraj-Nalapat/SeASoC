/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE675_Duplicate_Operations_on_Resource__w32CreateFile_66a.c
Label Definition File: CWE675_Duplicate_Operations_on_Resource__w32CreateFile.label.xml
Template File: sources-sinks-66a.tmpl.c
*/
/*
 * @description
 * CWE: 675 Duplicate Operations on Resource
 * BadSource:  Open and close a file using CreateFile() and CloseHandle()
 * GoodSource: Open a file using CreateFile()
 * Sinks:
 *    GoodSink: Do nothing
 *    BadSink : Close the file
 * Flow Variant: 66 Data flow: data passed in an array from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <windows.h>

#ifndef OMITBAD

/* bad function declaration */
void CWE675_Duplicate_Operations_on_Resource__w32CreateFile_66b_bad_sink(HANDLE data_array[]);

void CWE675_Duplicate_Operations_on_Resource__w32CreateFile_66_bad()
{
    HANDLE data;
    HANDLE data_array[5];
    data = INVALID_HANDLE_VALUE; /* Initialize data */
    data = CreateFile("BadSource_w32CreateFile.txt",
                      (GENERIC_WRITE|GENERIC_READ),
                      0,
                      NULL,
                      OPEN_ALWAYS,
                      FILE_ATTRIBUTE_NORMAL,
                      NULL);
    /* POTENTIAL FLAW: Close the file in the source */
    CloseHandle(data);
    /* put data in array */
    data_array[2] = data;
    CWE675_Duplicate_Operations_on_Resource__w32CreateFile_66b_bad_sink(data_array);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE675_Duplicate_Operations_on_Resource__w32CreateFile_66b_goodG2B_sink(HANDLE data_array[]);

static void goodG2B()
{
    HANDLE data;
    HANDLE data_array[5];
    data = INVALID_HANDLE_VALUE; /* Initialize data */
    /* FIX: Open, but do not close the file in the source */
    data = CreateFile("GoodSource_w32CreateFile.txt",
                      (GENERIC_WRITE|GENERIC_READ),
                      0,
                      NULL,
                      OPEN_ALWAYS,
                      FILE_ATTRIBUTE_NORMAL,
                      NULL);
    data_array[2] = data;
    CWE675_Duplicate_Operations_on_Resource__w32CreateFile_66b_goodG2B_sink(data_array);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE675_Duplicate_Operations_on_Resource__w32CreateFile_66b_goodB2G_sink(HANDLE data_array[]);

static void goodB2G()
{
    HANDLE data;
    HANDLE data_array[5];
    data = INVALID_HANDLE_VALUE; /* Initialize data */
    data = CreateFile("BadSource_w32CreateFile.txt",
                      (GENERIC_WRITE|GENERIC_READ),
                      0,
                      NULL,
                      OPEN_ALWAYS,
                      FILE_ATTRIBUTE_NORMAL,
                      NULL);
    /* POTENTIAL FLAW: Close the file in the source */
    CloseHandle(data);
    data_array[2] = data;
    CWE675_Duplicate_Operations_on_Resource__w32CreateFile_66b_goodB2G_sink(data_array);
}

void CWE675_Duplicate_Operations_on_Resource__w32CreateFile_66_good()
{
    goodG2B();
    goodB2G();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE675_Duplicate_Operations_on_Resource__w32CreateFile_66_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE675_Duplicate_Operations_on_Resource__w32CreateFile_66_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
