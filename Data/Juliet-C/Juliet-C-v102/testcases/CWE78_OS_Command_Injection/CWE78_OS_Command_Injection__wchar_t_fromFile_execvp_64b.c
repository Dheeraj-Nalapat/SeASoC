/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__wchar_t_fromFile_execvp_64b.c
Label Definition File: CWE78_OS_Command_Injection.no_path.label.xml
Template File: sources-sink-64b.tmpl.c
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: fromFile Read input from a file
 * GoodSource: Benign input
 * Sinks: execvp
 *    BadSink : execute command with wexecvp
 * Flow Variant: 64 Data flow: void pointer to data passed from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
# define COMMAND_INT_PATH L"%WINDIR%\\system32\\cmd.exe"
# define COMMAND_INT L"cmd.exe"
# define COMMAND_ARG1 L"/c"
# define COMMAND_ARG2 L"dir"
# define COMMAND_ARG3 data
#else /* NOT _WIN32 */
# define COMMAND_INT_PATH L"/bin/sh"
# define COMMAND_INT L"sh"
# define COMMAND_ARG1 L"ls"
# define COMMAND_ARG2 data
# define COMMAND_ARG3 NULL
#endif

#ifdef _WIN32
# define FOPEN _wfopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

#ifdef _WIN32
#include <process.h>
# define EXECVP _wexecvp
#else /* NOT _WIN32 */
# define EXECVP wexecvp
#endif

#ifndef OMITBAD

void CWE78_OS_Command_Injection__wchar_t_fromFile_execvp_64b_bad_sink(void * void_data_ptr)
{
    /* cast void pointer to a pointer of the appropriate type */
    wchar_t * * data_ptr = (wchar_t * *)void_data_ptr;
    /* dereference data_ptr into data */
    wchar_t * data = (*data_ptr);
    {
        wchar_t *args[] = {COMMAND_INT_PATH, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL};
        /* wexecvp - searches for the location of the command among
         * the directories specified by the PATH environment variable */
        /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
        EXECVP(COMMAND_INT, args);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE78_OS_Command_Injection__wchar_t_fromFile_execvp_64b_goodG2B_sink(void * void_data_ptr)
{
    /* cast void pointer to a pointer of the appropriate type */
    wchar_t * * data_ptr = (wchar_t * *)void_data_ptr;
    /* dereference data_ptr into data */
    wchar_t * data = (*data_ptr);
    {
        wchar_t *args[] = {COMMAND_INT_PATH, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL};
        /* wexecvp - searches for the location of the command among
         * the directories specified by the PATH environment variable */
        /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
        EXECVP(COMMAND_INT, args);
    }
}

#endif /* OMITGOOD */
