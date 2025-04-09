	.file	"demon90s_CppStudy_ex_09_flatten.c"
	.text
	.globl	_TIG_IZ_qObG_argc
	.bss
	.align 4
	.type	_TIG_IZ_qObG_argc, @object
	.size	_TIG_IZ_qObG_argc, 4
_TIG_IZ_qObG_argc:
	.zero	4
	.globl	_TIG_IZ_qObG_argv
	.align 8
	.type	_TIG_IZ_qObG_argv, @object
	.size	_TIG_IZ_qObG_argv, 8
_TIG_IZ_qObG_argv:
	.zero	8
	.globl	_TIG_IZ_qObG_envp
	.align 8
	.type	_TIG_IZ_qObG_envp, @object
	.size	_TIG_IZ_qObG_envp, 8
_TIG_IZ_qObG_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"folks!"
	.text
	.globl	f2
	.type	f2, @function
f2:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L7
	cmpq	$1, -8(%rbp)
	jne	.L8
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L4
.L8:
	nop
.L4:
	jmp	.L6
.L7:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	f2, .-f2
	.section	.rodata
.LC1:
	.string	"That's all, "
	.text
	.globl	f1
	.type	f1, @function
f1:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L14:
	cmpq	$0, -8(%rbp)
	je	.L10
	cmpq	$1, -8(%rbp)
	jne	.L16
	jmp	.L15
.L10:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L13
.L16:
	nop
.L13:
	jmp	.L14
.L15:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	f1, .-f1
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_qObG_envp(%rip)
	nop
.L18:
	movq	$0, _TIG_IZ_qObG_argv(%rip)
	nop
.L19:
	movl	$0, _TIG_IZ_qObG_argc(%rip)
	nop
	nop
.L20:
.L21:
#APP
# 59 "demon90s_CppStudy_ex_09.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-qObG--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_qObG_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_qObG_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_qObG_envp(%rip)
	nop
	movq	$2, -8(%rbp)
.L27:
	cmpq	$2, -8(%rbp)
	je	.L22
	cmpq	$2, -8(%rbp)
	ja	.L29
	cmpq	$0, -8(%rbp)
	je	.L24
	cmpq	$1, -8(%rbp)
	jne	.L29
	movl	$0, %eax
	jmp	.L28
.L24:
	leaq	f2(%rip), %rax
	movq	%rax, %rdi
	call	atexit@PLT
	leaq	f1(%rip), %rax
	movq	%rax, %rdi
	call	atexit@PLT
	movq	$1, -8(%rbp)
	jmp	.L26
.L22:
	movq	$0, -8(%rbp)
	jmp	.L26
.L29:
	nop
.L26:
	jmp	.L27
.L28:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
