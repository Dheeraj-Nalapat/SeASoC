/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE321_Hard_Coded_Cryptographic_Key__w32_char_61b.c
Label Definition File: CWE321_Hard_Coded_Cryptographic_Key__w32.label.xml
Template File: sources-sink-61b.tmpl.c
*/
/*
 * @description
 * CWE: 321 Use of Hard-coded Cryptographic Key
 * BadSource: hardcoded Copy a hardcoded value into cryptokey
 * GoodSource: Read cryptokey from the console
 * Sinks:
 *    BadSink : Hash cryptokey and use the value to encrypt a string
 * Flow Variant: 61 Data flow: data returned from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#define CRYPTOKEY "Hardcoded"

#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>

/* Link with the Advapi32.lib file for Crypt* functions */
#pragma comment (lib, "Advapi32")
#endif

#ifndef OMITBAD

char * CWE321_Hard_Coded_Cryptographic_Key__w32_char_61b_bad_source(char * cryptokey)
{
    /* FLAW: Use a hardcoded value for the hash input causing a hardcoded crypto key in the sink */
    strcpy(cryptokey, CRYPTOKEY);
    return cryptokey;
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
char * CWE321_Hard_Coded_Cryptographic_Key__w32_char_61b_goodG2B_source(char * cryptokey)
{
    {
        size_t cryptokey_len = strlen(cryptokey);
        /* if there is room in cryptokey, read into it from the console */
        if(100-cryptokey_len > 1)
        {
            /* FIX: Obtain the hash input from the console */
            fgets(cryptokey+cryptokey_len, (int)(100-cryptokey_len), stdin);
            /* The next 3 lines remove the carriage return from the string that is
             * inserted by fgets() */
            cryptokey_len = strlen(cryptokey);
            if (cryptokey_len > 0)
            {
                cryptokey[cryptokey_len-1] = '\0';
            }
        }
    }
    return cryptokey;
}

#endif /* OMITGOOD */
