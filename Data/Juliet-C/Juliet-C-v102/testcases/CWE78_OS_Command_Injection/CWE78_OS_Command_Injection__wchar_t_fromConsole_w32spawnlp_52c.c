/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__wchar_t_fromConsole_w32spawnlp_52c.c
Label Definition File: CWE78_OS_Command_Injection.no_path.label.xml
Template File: sources-sink-52c.tmpl.c
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: fromConsole Read input from the console
 * GoodSource: Benign input
 * Sink: w32spawnlp
 *    BadSink : execute command with wspawnlp
 * Flow Variant: 52 Data flow: data passed as an argument from one function to another to another in three different source files
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

#include <process.h>

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

void CWE78_OS_Command_Injection__wchar_t_fromConsole_w32spawnlp_52c_bad_sink(wchar_t * data)
{
    /* wspawnlp - searches for the location of the command among
     * the directories specified by the PATH environment variable */
    /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
    _wspawnlp(_P_WAIT, COMMAND_INT, COMMAND_INT, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE78_OS_Command_Injection__wchar_t_fromConsole_w32spawnlp_52c_goodG2B_sink(wchar_t * data)
{
    /* wspawnlp - searches for the location of the command among
     * the directories specified by the PATH environment variable */
    /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
    _wspawnlp(_P_WAIT, COMMAND_INT, COMMAND_INT, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL);
}

#endif /* OMITGOOD */
