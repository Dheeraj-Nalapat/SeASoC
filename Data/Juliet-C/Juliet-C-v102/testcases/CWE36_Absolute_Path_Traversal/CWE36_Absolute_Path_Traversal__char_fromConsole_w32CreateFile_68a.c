/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE36_Absolute_Path_Traversal__char_fromConsole_w32CreateFile_68a.c
Label Definition File: CWE36_Absolute_Path_Traversal.label.xml
Template File: sources-sink-68a.tmpl.c
*/
/*
 * @description
 * CWE: 36 Absolute Path Traversal
 * BadSource: fromConsole Read input from the console
 * GoodSource: Full path and file name
 * Sink: w32CreateFile
 *    BadSink :
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <windows.h>

char * CWE36_Absolute_Path_Traversal__char_fromConsole_w32CreateFile_68_bad_data;
char * CWE36_Absolute_Path_Traversal__char_fromConsole_w32CreateFile_68_goodG2B_data;

#ifndef OMITBAD

/* bad function declaration */
void CWE36_Absolute_Path_Traversal__char_fromConsole_w32CreateFile_68b_bad_sink();

void CWE36_Absolute_Path_Traversal__char_fromConsole_w32CreateFile_68_bad()
{
    char * data;
    char data_buf[FILENAME_MAX] = "";
    data = data_buf;
    {
        /* Read input from the console */
        size_t data_len = strlen(data);
        /* if there is room in data, read into it from the console */
        if(FILENAME_MAX-data_len > 1)
        {
            fgets(data+data_len, (int)(FILENAME_MAX-data_len), stdin);
            /* The next 3 lines remove the carriage return from the string that is
             * inserted by fgets() */
            data_len = strlen(data);
            if (data_len > 0)
            {
                data[data_len-1] = '\0';
            }
        }
    }
    CWE36_Absolute_Path_Traversal__char_fromConsole_w32CreateFile_68_bad_data = data;
    CWE36_Absolute_Path_Traversal__char_fromConsole_w32CreateFile_68b_bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declarations */
void CWE36_Absolute_Path_Traversal__char_fromConsole_w32CreateFile_68b_goodG2B_sink();

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * data;
    char data_buf[FILENAME_MAX] = "";
    data = data_buf;
#ifdef _WIN32
    /* FIX: Full path and file name */
    strcpy(data, "c:\\temp\\file.txt");
#else
    /* FIX: Full path and file name */
    strcpy(data, "/tmp/file.txt");
#endif
    CWE36_Absolute_Path_Traversal__char_fromConsole_w32CreateFile_68_goodG2B_data = data;
    CWE36_Absolute_Path_Traversal__char_fromConsole_w32CreateFile_68b_goodG2B_sink();
}

void CWE36_Absolute_Path_Traversal__char_fromConsole_w32CreateFile_68_good()
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
    CWE36_Absolute_Path_Traversal__char_fromConsole_w32CreateFile_68_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE36_Absolute_Path_Traversal__char_fromConsole_w32CreateFile_68_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
