/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__char_fromConsole_system_54c.c
Label Definition File: CWE78_OS_Command_Injection.fullpath.label.xml
Template File: sources-sink-54c.tmpl.c
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: fromConsole Read input from the console
 * GoodSource: Benign input
 * Sink: system
 *    BadSink : Execute command using system()
 * Flow Variant: 54 Data flow: data passed as an argument from one function through three others to a fifth; all five functions are in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
# define COMMAND_INT_PATH "%WINDIR%\\system32\\cmd.exe"
# define COMMAND_INT "cmd.exe"
# define COMMAND_ARG1 "/c"
# define COMMAND_ARG2 "dir"
# define FULL_COMMAND COMMAND_INT_PATH " " COMMAND_ARG1 " "COMMAND_ARG2 " "
#else /* NOT _WIN32 */
# define COMMAND_INT_PATH "/bin/sh"
# define COMMAND_INT "sh"
# define COMMAND_ARG1 "ls"
# define FULL_COMMAND COMMAND_INT_PATH " " COMMAND_ARG1 " "
#endif

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

/* bad function declaration */
void CWE78_OS_Command_Injection__char_fromConsole_system_54d_bad_sink(char * data);

void CWE78_OS_Command_Injection__char_fromConsole_system_54c_bad_sink(char * data)
{
    CWE78_OS_Command_Injection__char_fromConsole_system_54d_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declaration */
void CWE78_OS_Command_Injection__char_fromConsole_system_54d_goodG2B_sink(char * data);

/* goodG2B uses the GoodSource with the BadSink */
void CWE78_OS_Command_Injection__char_fromConsole_system_54c_goodG2B_sink(char * data)
{
    CWE78_OS_Command_Injection__char_fromConsole_system_54d_goodG2B_sink(data);
}

#endif /* OMITGOOD */
