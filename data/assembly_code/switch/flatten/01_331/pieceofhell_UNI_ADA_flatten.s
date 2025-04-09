	.file	"pieceofhell_UNI_ADA_flatten.c"
	.text
	.globl	_TIG_IZ_8Dbx_argv
	.bss
	.align 8
	.type	_TIG_IZ_8Dbx_argv, @object
	.size	_TIG_IZ_8Dbx_argv, 8
_TIG_IZ_8Dbx_argv:
	.zero	8
	.globl	_TIG_IZ_8Dbx_argc
	.align 4
	.type	_TIG_IZ_8Dbx_argc, @object
	.size	_TIG_IZ_8Dbx_argc, 4
_TIG_IZ_8Dbx_argc:
	.zero	4
	.globl	_TIG_IZ_8Dbx_envp
	.align 8
	.type	_TIG_IZ_8Dbx_envp, @object
	.size	_TIG_IZ_8Dbx_envp, 8
_TIG_IZ_8Dbx_envp:
	.zero	8
	.text
	.globl	misterio
	.type	misterio, @function
misterio:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$8, -8(%rbp)
.L20:
	cmpq	$9, -8(%rbp)
	ja	.L21
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L21-.L4
	.long	.L21-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L7:
	movl	-20(%rbp), %eax
	jmp	.L12
.L5:
	cmpl	$3, -20(%rbp)
	je	.L13
	cmpl	$3, -20(%rbp)
	jg	.L14
	cmpl	$2, -20(%rbp)
	je	.L15
	cmpl	$2, -20(%rbp)
	jg	.L14
	cmpl	$0, -20(%rbp)
	je	.L16
	cmpl	$1, -20(%rbp)
	je	.L17
	jmp	.L14
.L13:
	movq	$1, -8(%rbp)
	jmp	.L18
.L15:
	movq	$9, -8(%rbp)
	jmp	.L18
.L17:
	movq	$3, -8(%rbp)
	jmp	.L18
.L16:
	movq	$4, -8(%rbp)
	jmp	.L18
.L14:
	movq	$0, -8(%rbp)
	nop
.L18:
	jmp	.L19
.L10:
	movl	-20(%rbp), %edx
	movl	%edx, %eax
	addl	%eax, %eax
	addl	%edx, %eax
	movl	%eax, -20(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L19
.L8:
	movl	-20(%rbp), %eax
	imull	%eax, %eax
	movl	%eax, -20(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L19
.L3:
	sall	-20(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L19
.L6:
	movl	-20(%rbp), %eax
	cltd
	shrl	$30, %edx
	addl	%edx, %eax
	andl	$3, %eax
	subl	%edx, %eax
	movl	%eax, %edi
	call	misterio
	movl	%eax, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L19
.L11:
	sall	$2, -20(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L19
.L9:
	movl	-12(%rbp), %eax
	jmp	.L12
.L21:
	nop
.L19:
	jmp	.L20
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	misterio, .-misterio
	.section	.rodata
.LC0:
	.string	"Ol\303\241 Mundo!"
.LC1:
	.string	"%d\n"
	.text
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$0, _TIG_IZ_8Dbx_envp(%rip)
	nop
.L23:
	movq	$0, _TIG_IZ_8Dbx_argv(%rip)
	nop
.L24:
	movl	$0, _TIG_IZ_8Dbx_argc(%rip)
	nop
	nop
.L25:
.L26:
#APP
# 74 "pieceofhell_UNI_ADA.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-8Dbx--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_8Dbx_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_8Dbx_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_8Dbx_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L39:
	cmpq	$6, -8(%rbp)
	ja	.L41
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L29(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L29(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L29:
	.long	.L34-.L29
	.long	.L33-.L29
	.long	.L32-.L29
	.long	.L41-.L29
	.long	.L31-.L29
	.long	.L30-.L29
	.long	.L28-.L29
	.text
.L31:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -8(%rbp)
	jmp	.L35
.L33:
	movq	$2, -8(%rbp)
	jmp	.L35
.L28:
	movl	$0, %eax
	jmp	.L40
.L30:
	movl	$3, %edi
	call	misterio
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -8(%rbp)
	jmp	.L35
.L34:
	movl	-32(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1431655766, %rax, %rax
	shrq	$32, %rax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, %ecx
	addl	%ecx, %ecx
	addl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	je	.L37
	movq	$4, -8(%rbp)
	jmp	.L35
.L37:
	movq	$5, -8(%rbp)
	jmp	.L35
.L32:
	movl	$4, -24(%rbp)
	movl	$6, -20(%rbp)
	movl	$9, -32(%rbp)
	movl	$0, -16(%rbp)
	movl	$1, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L35
.L41:
	nop
.L35:
	jmp	.L39
.L40:
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
