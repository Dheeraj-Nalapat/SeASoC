/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__wchar_t_fromFile_system_63b.c
Label Definition File: CWE78_OS_Command_Injection.fullpath.label.xml
Template File: sources-sink-63b.tmpl.c
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: fromFile Read input from a file
 * GoodSource: Benign input
 * Sinks: system
 *    BadSink : Execute command using system()
 * Flow Variant: 63 Data flow: pointer to data passed from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
# define COMMAND_INT_PATH L"%WINDIR%\\system32\\cmd.exe"
# define COMMAND_INT L"cmd.exe"
# define COMMAND_ARG1 L"/c"
# define COMMAND_ARG2 L"dir"
# define FULL_COMMAND COMMAND_INT_PATH L" " COMMAND_ARG1 L" "COMMAND_ARG2 L" "
#else /* NOT _WIN32 */
# define COMMAND_INT_PATH L"/bin/sh"
# define COMMAND_INT L"sh"
# define COMMAND_ARG1 L"ls"
# define FULL_COMMAND COMMAND_INT_PATH L" " COMMAND_ARG1 L" "
#endif

#ifdef _WIN32
# define FOPEN _wfopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

#ifndef OMITBAD

void CWE78_OS_Command_Injection__wchar_t_fromFile_system_63b_bad_sink(wchar_t * * data_ptr)
{
    wchar_t * data = *data_ptr;
    /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
    _wsystem(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE78_OS_Command_Injection__wchar_t_fromFile_system_63b_goodG2B_sink(wchar_t * * data_ptr)
{
    wchar_t * data = *data_ptr;
    /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
    _wsystem(data);
}

#endif /* OMITGOOD */
