/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE23_Relative_Path_Traversal__char_fromFile_w32CreateFile_61a.c
Label Definition File: CWE23_Relative_Path_Traversal.label.xml
Template File: sources-sink-61a.tmpl.c
*/
/*
 * @description
 * CWE: 23 Relative Path Traversal
 * BadSource: fromFile Read input from a file
 * GoodSource: File name without a period or slash
 * Sinks: w32CreateFile
 *    BadSink :
 * Flow Variant: 61 Data flow: data returned from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
#define BASEPATH "c:\\temp\\"
#else
#define BASEPATH "/tmp/"
#endif

#ifdef _WIN32
# define FOPEN fopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

#include <windows.h>

#ifndef OMITBAD

/* bad function declaration */
char * CWE23_Relative_Path_Traversal__char_fromFile_w32CreateFile_61b_bad_source(char * data);

void CWE23_Relative_Path_Traversal__char_fromFile_w32CreateFile_61_bad()
{
    char * data;
    char data_buf[FILENAME_MAX] = BASEPATH;
    data = data_buf;
    data = CWE23_Relative_Path_Traversal__char_fromFile_w32CreateFile_61b_bad_source(data);
    {
        HANDLE hFile;
        /* POTENTIAL FLAW: Possibly creating and opening a file without validating the file name or path */
        hFile = CreateFileA(data,
                            (GENERIC_WRITE|GENERIC_READ),
                            0,
                            NULL,
                            OPEN_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL,
                            NULL);
        if (hFile != INVALID_HANDLE_VALUE)
        {
            CloseHandle(hFile);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
char * CWE23_Relative_Path_Traversal__char_fromFile_w32CreateFile_61b_goodG2B_source(char * data);

static void goodG2B()
{
    char * data;
    char data_buf[FILENAME_MAX] = BASEPATH;
    data = data_buf;
    data = CWE23_Relative_Path_Traversal__char_fromFile_w32CreateFile_61b_goodG2B_source(data);
    {
        HANDLE hFile;
        /* POTENTIAL FLAW: Possibly creating and opening a file without validating the file name or path */
        hFile = CreateFileA(data,
                            (GENERIC_WRITE|GENERIC_READ),
                            0,
                            NULL,
                            OPEN_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL,
                            NULL);
        if (hFile != INVALID_HANDLE_VALUE)
        {
            CloseHandle(hFile);
        }
    }
}

void CWE23_Relative_Path_Traversal__char_fromFile_w32CreateFile_61_good()
{
    goodG2B();
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
    CWE23_Relative_Path_Traversal__char_fromFile_w32CreateFile_61_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE23_Relative_Path_Traversal__char_fromFile_w32CreateFile_61_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
