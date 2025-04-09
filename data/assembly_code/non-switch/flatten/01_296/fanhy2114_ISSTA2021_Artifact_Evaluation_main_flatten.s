	.file	"fanhy2114_ISSTA2021_Artifact_Evaluation_main_flatten.c"
	.text
	.globl	_TIG_IZ_p4GM_argc
	.bss
	.align 4
	.type	_TIG_IZ_p4GM_argc, @object
	.size	_TIG_IZ_p4GM_argc, 4
_TIG_IZ_p4GM_argc:
	.zero	4
	.globl	_TIG_IZ_p4GM_argv
	.align 8
	.type	_TIG_IZ_p4GM_argv, @object
	.size	_TIG_IZ_p4GM_argv, 8
_TIG_IZ_p4GM_argv:
	.zero	8
	.globl	_TIG_IZ_p4GM_envp
	.align 8
	.type	_TIG_IZ_p4GM_envp, @object
	.size	_TIG_IZ_p4GM_envp, 8
_TIG_IZ_p4GM_envp:
	.zero	8
	.globl	x
	.align 4
	.type	x, @object
	.size	x, 4
x:
	.zero	4
	.text
	.globl	func3
	.type	func3, @function
func3:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L7
	cmpq	$1, -8(%rbp)
	jne	.L8
	movl	$6, x(%rip)
	movq	$0, -8(%rbp)
	jmp	.L4
.L8:
	nop
.L4:
	jmp	.L6
.L7:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	func3, .-func3
	.globl	func1
	.type	func1, @function
func1:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L15:
	cmpq	$2, -8(%rbp)
	je	.L10
	cmpq	$2, -8(%rbp)
	ja	.L16
	cmpq	$0, -8(%rbp)
	je	.L17
	cmpq	$1, -8(%rbp)
	jne	.L16
	movl	$1, x(%rip)
	call	func2
	movl	$2, x(%rip)
	movq	$0, -8(%rbp)
	jmp	.L13
.L10:
	movq	$1, -8(%rbp)
	jmp	.L13
.L16:
	nop
.L13:
	jmp	.L15
.L17:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	func1, .-func1
	.globl	main
	.type	main, @function
main:
.LFB8:
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
	movl	$0, x(%rip)
	nop
.L19:
	movq	$0, _TIG_IZ_p4GM_envp(%rip)
	nop
.L20:
	movq	$0, _TIG_IZ_p4GM_argv(%rip)
	nop
.L21:
	movl	$0, _TIG_IZ_p4GM_argc(%rip)
	nop
	nop
.L22:
.L23:
#APP
# 103 "fanhy2114_ISSTA2021_Artifact_Evaluation_main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-p4GM--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_p4GM_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_p4GM_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_p4GM_envp(%rip)
	nop
	movq	$3, -8(%rbp)
.L29:
	cmpq	$3, -8(%rbp)
	je	.L24
	cmpq	$3, -8(%rbp)
	ja	.L31
	cmpq	$0, -8(%rbp)
	je	.L26
	cmpq	$1, -8(%rbp)
	jne	.L31
	call	func1
	movl	$5, x(%rip)
	movq	$0, -8(%rbp)
	jmp	.L27
.L24:
	movl	$4, x(%rip)
	movq	$1, -8(%rbp)
	jmp	.L27
.L26:
	movl	$0, %eax
	jmp	.L30
.L31:
	nop
.L27:
	jmp	.L29
.L30:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	main, .-main
	.globl	func2
	.type	func2, @function
func2:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L37:
	cmpq	$0, -8(%rbp)
	je	.L38
	cmpq	$1, -8(%rbp)
	jne	.L39
	movl	$3, x(%rip)
	movq	$0, -8(%rbp)
	jmp	.L35
.L39:
	nop
.L35:
	jmp	.L37
.L38:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	func2, .-func2
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
