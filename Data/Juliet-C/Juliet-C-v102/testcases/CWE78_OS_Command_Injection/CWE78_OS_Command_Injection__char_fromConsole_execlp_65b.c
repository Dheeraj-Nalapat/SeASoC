/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__char_fromConsole_execlp_65b.c
Label Definition File: CWE78_OS_Command_Injection.no_path.label.xml
Template File: sources-sink-65b.tmpl.c
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: fromConsole Read input from the console
 * GoodSource: Benign input
 * Sinks: execlp
 *    BadSink : execute command with execlp
 * Flow Variant: 65 Data/control flow: data passed as an argument from one function to a function in a different source file called via a function pointer
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
#include <process.h>
# define EXECLP _execlp
#else /* NOT _WIN32 */
# define EXECLP execlp
#endif

#ifndef OMITBAD

void CWE78_OS_Command_Injection__char_fromConsole_execlp_65b_bad_sink(char * data)
{
    /* execlp - searches for the location of the command among
     * the directories specified by the PATH environment variable */
    /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
    EXECLP(COMMAND_INT, COMMAND_INT, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE78_OS_Command_Injection__char_fromConsole_execlp_65b_goodG2B_sink(char * data)
{
    /* execlp - searches for the location of the command among
     * the directories specified by the PATH environment variable */
    /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
    EXECLP(COMMAND_INT, COMMAND_INT, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL);
}

#endif /* OMITGOOD */
