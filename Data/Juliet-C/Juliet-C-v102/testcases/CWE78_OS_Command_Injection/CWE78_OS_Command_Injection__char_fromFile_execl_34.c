/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__char_fromFile_execl_34.c
Label Definition File: CWE78_OS_Command_Injection.no_path.label.xml
Template File: sources-sink-34.tmpl.c
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: fromFile Read input from a file
 * GoodSource: Benign input
 * Sinks: execl
 *    BadSink : execute command with execl
 * Flow Variant: 34 Data flow: use of a union containing two methods of accessing the same data (within the same function)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
# define COMMAND_INT_PATH "%WINDIR%\\system32\\cmd.exe"
# define COMMAND_INT "cmd.exe"
# define COMMAND_ARG1 "/c"
# define COMMAND_ARG2 "dir"
# define COMMAND_ARG3 data
#else /* NOT _WIN32 */
# define COMMAND_INT_PATH "/bin/sh"
# define COMMAND_INT "sh"
# define COMMAND_ARG1 "ls"
# define COMMAND_ARG2 data
# define COMMAND_ARG3 NULL
#endif

#ifdef _WIN32
# define FOPEN fopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

#ifdef _WIN32
#include <process.h>
# define EXECL _execl
#else /* NOT _WIN32 */
# define EXECL execl
#endif

typedef union
{
    char * a;
    char * b;
} CWE78_OS_Command_Injection__char_fromFile_execl_34_union_type;

#ifndef OMITBAD

void CWE78_OS_Command_Injection__char_fromFile_execl_34_bad()
{
    char * data;
    CWE78_OS_Command_Injection__char_fromFile_execl_34_union_type my_union;
    char data_buf[100] = "";
    data = data_buf;
    {
        /* Read input from a file */
        size_t data_len = strlen(data);
        FILE * pFile;
        /* if there is room in data, attempt to read the input from a file */
        if(100-data_len > 1)
        {
            pFile = FOPEN("C:\\temp\\file.txt", "r");
            if (pFile != NULL)
            {
                fgets(data+data_len, (int)(100-data_len), pFile);
                fclose(pFile);
            }
        }
    }
    my_union.a = data;
    {
        char * data = my_union.b;
        /* execl - specify the path where the command is located */
        /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
        EXECL(COMMAND_INT_PATH, COMMAND_INT_PATH, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * data;
    CWE78_OS_Command_Injection__char_fromFile_execl_34_union_type my_union;
    char data_buf[100] = "";
    data = data_buf;
    /* FIX: Benign input preventing command injection */
    strcat(data, "*.*");
    my_union.a = data;
    {
        char * data = my_union.b;
        /* execl - specify the path where the command is located */
        /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
        EXECL(COMMAND_INT_PATH, COMMAND_INT_PATH, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL);
    }
}

void CWE78_OS_Command_Injection__char_fromFile_execl_34_good()
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
    CWE78_OS_Command_Injection__char_fromFile_execl_34_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE78_OS_Command_Injection__char_fromFile_execl_34_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
